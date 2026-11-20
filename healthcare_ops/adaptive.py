"""Rolling 56-day strategy backtest for synthetic bed and nurse planning."""

from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

from allocation import solve_allocation_ilp
from config import (
    ADAPTIVE_BACKTEST_DAYS,
    ADAPTIVE_RETRAIN_INTERVAL_DAYS,
    DEPARTMENTS,
    DEFAULT_TOTAL_BEDS,
    DEFAULT_TOTAL_NURSES,
    FORECAST_SAFETY_MARGIN,
    NURSE_PATIENT_RATIOS,
    RESULTS_DIR,
    URGENCY_WEIGHTS,
)
from forecasting import fit_forecasters, forecast_next_day


STRATEGIES = (
    "static_28day_average",
    "naive_last_week",
    "forecast_without_retraining",
    "forecast_with_weekly_retraining",
    "forecast_plus_safety_margin",
    "proportional_split",
    "oracle",
)


def _validate(data: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "department", "census", "outbreak"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Backtest input is missing: {sorted(missing)}")
    result = data.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["census"] = pd.to_numeric(result["census"], errors="raise")
    if result[["date", "department"]].duplicated().any():
        raise ValueError("Backtest needs one row per department and date.")
    return result.sort_values(["department", "date"]).reset_index(drop=True)


def _history_on_date(
    data: pd.DataFrame,
    department: str,
    date: pd.Timestamp,
) -> pd.DataFrame:
    history = data.loc[
        (data["department"] == department)
        & (data["date"] <= date)
    ].sort_values("date")
    if history.empty:
        raise ValueError(f"No history for {department}.")
    return history


def _naive_demands(
    data: pd.DataFrame,
    date: pd.Timestamp,
) -> dict[str, float]:
    """Use each department's census exactly seven days earlier."""
    demands: dict[str, float] = {}
    for department in DEPARTMENTS:
        row = data.loc[
            (data["department"] == department)
            & (data["date"] == date - pd.Timedelta(days=7))
        ]
        if len(row) != 1:
            raise ValueError(
                f"Missing naive baseline census for {department} on "
                f"{(date - pd.Timedelta(days=7)).date()}."
            )
        demands[department] = float(row["census"].iloc[0])
    return demands


def _static_demands(
    data: pd.DataFrame,
    date: pd.Timestamp,
) -> dict[str, float]:
    """Use a trailing 28-day census average for each department."""
    demands: dict[str, float] = {}
    start = date - pd.Timedelta(days=28)
    for department in DEPARTMENTS:
        history = data.loc[
            (data["department"] == department)
            & (data["date"] > start)
            & (data["date"] <= date)
        ]
        if len(history) != 28:
            raise ValueError(
                f"Static baseline needs 28 prior rows for {department}."
            )
        demands[department] = float(history["census"].mean())
    return demands


def _forecast_demands(
    data: pd.DataFrame,
    date: pd.Timestamp,
    bundle,
) -> dict[str, float]:
    """Return fitted one-day-ahead census forecasts for the next date."""
    outlook = forecast_next_day(data, as_of=date, bundle=bundle)
    target = date + pd.Timedelta(days=1)
    demands: dict[str, float] = {}
    for department in DEPARTMENTS:
        row = outlook.loc[
            (outlook["department"] == department)
            & (outlook["date"] == target)
        ]
        if len(row) != 1:
            raise ValueError(
                f"Missing forecast for {department} on {target.date()}."
            )
        demands[department] = float(row["forecast_census"].iloc[0])
    return demands


def _oracle_demands(data: pd.DataFrame, date: pd.Timestamp) -> dict[str, float]:
    """True next-day census; used only as an optimistic benchmark."""
    target = date + pd.Timedelta(days=1)
    return {
        department: float(
            data.loc[
                (data["department"] == department)
                & (data["date"] == target),
                "census",
            ].iloc[0]
        )
        for department in DEPARTMENTS
    }


def proportional_split(
    demand: dict[str, float],
    total_beds: int,
    total_nurses: int,
) -> dict[str, dict[str, int]]:
    """Allocate beds and nurse capacity proportionally, with no optimizer."""
    return _proportional_split(demand, total_beds, total_nurses)


def _proportional_split(
    demand: dict[str, float],
    total_beds: int,
    total_nurses: int,
) -> dict[str, dict[str, int]]:
    """Two-stage largest-remainder split of beds and nurse capacity."""
    weights = {department: max(0.0, demand[department]) for department in DEPARTMENTS}
    total_weight = sum(weights.values())
    beds = {department: 0 for department in DEPARTMENTS}
    nurses = {department: 0 for department in DEPARTMENTS}

    if total_weight > 0:
        exact_beds = {
            department: total_beds * weights[department] / total_weight
            for department in DEPARTMENTS
        }
        for department in DEPARTMENTS:
            beds[department] = int(np.floor(exact_beds[department]))
        ranked = sorted(
            DEPARTMENTS,
            key=lambda department: (
                -(exact_beds[department] - beds[department]),
                DEPARTMENTS.index(department),
            ),
        )
        for department in ranked[: max(0, total_beds - sum(beds.values()))]:
            beds[department] += 1

    # A nurse allocation is based on the bed allocation divided by ratio.
    nurse_required = {
        department: ceil(beds[department] / NURSE_PATIENT_RATIOS[department])
        for department in DEPARTMENTS
    }
    if sum(nurse_required.values()) <= total_nurses:
        nurses = nurse_required
        remaining_nurses = total_nurses - sum(nurses.values())
        while remaining_nurses:
            unmet = {
                department: max(0.0, demand[department] - beds[department])
                for department in DEPARTMENTS
            }
            if sum(unmet.values()) == 0:
                break
            department = max(
                DEPARTMENTS,
                key=lambda name: (
                    unmet[name] / max(1.0, demand[name]),
                    -DEPARTMENTS.index(name),
                ),
            )
            nurses[department] += 1
            remaining_nurses -= 1
    else:
        nurse_weight = {
            department: max(0.0, demand[department])
            for department in DEPARTMENTS
        }
        total_nurse_weight = sum(nurse_weight.values())
        if total_nurse_weight:
            exact = {
                department: total_nurses * nurse_weight[department]
                / total_nurse_weight
                for department in DEPARTMENTS
            }
            for department in DEPARTMENTS:
                nurses[department] = int(np.floor(exact[department]))
            remaining = total_nurses - sum(nurses.values())
            ranked = sorted(
                DEPARTMENTS,
                key=lambda department: (
                    -(exact[department] - nurses[department]),
                    DEPARTMENTS.index(department),
                ),
            )
            for department in ranked[:remaining]:
                nurses[department] += 1
        beds = {
            department: min(
                beds[department],
                nurses[department] * NURSE_PATIENT_RATIOS[department],
            )
            for department in DEPARTMENTS
        }

    return {"patients": beds, "nurses": nurses}


def _score_allocation(
    allocated: dict[str, int],
    real_demand: dict[str, float],
) -> dict[str, float]:
    """Calculate weighted service and unmet patient totals for one day."""
    served = {
        department: float(
            min(
                max(0, allocated[department]),
                real_demand[department],
            )
        )
        for department in DEPARTMENTS
    }
    unmet = {
        department: float(max(0.0, real_demand[department] - served[department]))
        for department in DEPARTMENTS
    }
    weighted_served = sum(
        URGENCY_WEIGHTS[department] * served[department]
        for department in DEPARTMENTS
    )
    weighted_demand = sum(
        URGENCY_WEIGHTS[department] * real_demand[department]
        for department in DEPARTMENTS
    )
    return {
        "weighted_service_level": (
            weighted_served / weighted_demand if weighted_demand > 0 else 1.0
        ),
        "unmet_patients": sum(unmet.values()),
        "unmet_emergency_patients": unmet["Emergency"],
    }


def rolling_backtest(
    data: pd.DataFrame,
    days: int = ADAPTIVE_BACKTEST_DAYS,
    total_beds: int = DEFAULT_TOTAL_BEDS,
    total_nurses: int = DEFAULT_TOTAL_NURSES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all seven strategies across the latest days in the synthetic data."""
    if days < 1:
        raise ValueError("days must be positive.")
    data = _validate(data)
    last_date = data["date"].max()
    first_target = last_date - pd.Timedelta(days=days - 1)
    first_origin = first_target - pd.Timedelta(days=1)
    initial_model_data = data.loc[data["date"] <= first_origin]

    fixed_bundle = fit_forecasters(
        initial_model_data,
        train_until=first_origin,
        horizons=(1,),
    )
    weekly_bundle = fixed_bundle
    daily_rows: list[dict[str, object]] = []

    for date_index, origin in enumerate(
        pd.date_range(first_origin, last_date - pd.Timedelta(days=1), freq="D")
    ):
        target = origin + pd.Timedelta(days=1)
        real_demand = {
            department: float(
                data.loc[
                    (data["department"] == department)
                    & (data["date"] == target),
                    "census",
                ].iloc[0]
            )
            for department in DEPARTMENTS
        }
        outbreak = bool(
            data.loc[
                (data["department"] == "Emergency")
                & (data["date"] == target),
                "outbreak",
            ].iloc[0]
        )

        static_demands = _static_demands(data, origin)
        naive_demands = _naive_demands(data, origin)
        forecast_demands = _forecast_demands(data, origin, fixed_bundle)
        if date_index % ADAPTIVE_RETRAIN_INTERVAL_DAYS == 0:
            known_data = data.loc[data["date"] <= origin]
            weekly_bundle = fit_forecasters(
                known_data,
                train_until=origin,
                horizons=(1,),
            )
        retrained_demands = _forecast_demands(data, origin, weekly_bundle)

        planned = {}
        planned["static_28day_average"] = solve_allocation_ilp(
            static_demands, total_beds, total_nurses
        ).patients_served
        planned["naive_last_week"] = solve_allocation_ilp(
            naive_demands, total_beds, total_nurses
        ).patients_served
        planned["forecast_without_retraining"] = solve_allocation_ilp(
            forecast_demands, total_beds, total_nurses
        ).patients_served
        planned["forecast_with_weekly_retraining"] = solve_allocation_ilp(
            retrained_demands, total_beds, total_nurses
        ).patients_served
        planned["forecast_plus_safety_margin"] = solve_allocation_ilp(
            {
                department: demand * (1.0 + FORECAST_SAFETY_MARGIN)
                for department, demand in forecast_demands.items()
            },
            total_beds,
            total_nurses,
        ).patients_served
        planned["proportional_split"] = proportional_split(
            forecast_demands, total_beds, total_nurses
        )["patients"]
        planned["oracle"] = solve_allocation_ilp(
            real_demand, total_beds, total_nurses
        ).patients_served

        for strategy in STRATEGIES:
            score = _score_allocation(planned[strategy], real_demand)
            daily_rows.append(
                {
                    "date": target,
                    "strategy": strategy,
                    "period": "outbreak" if outbreak else "normal",
                    "weighted_service_level": score["weighted_service_level"],
                    "unmet_patients": score["unmet_patients"],
                    "unmet_emergency_patients": score[
                        "unmet_emergency_patients"
                    ],
                    "data_type": "synthetic",
                }
            )

    daily = pd.DataFrame(daily_rows)
    summary = (
        daily.groupby(["strategy", "period"], as_index=False)
        .agg(
            weighted_service_level=("weighted_service_level", "mean"),
            unmet_patients=("unmet_patients", "sum"),
            unmet_emergency_patients=("unmet_emergency_patients", "sum"),
        )
        .sort_values(["period", "strategy"])
        .reset_index(drop=True)
    )
    summary["data_type"] = "synthetic"
    return summary, daily


def main() -> None:
    """Run and save the synthetic adaptive backtest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "synthetic_daily_data.csv",
    )
    parser.add_argument("--days", type=int, default=ADAPTIVE_BACKTEST_DAYS)
    args = parser.parse_args()

    data = pd.read_csv(args.data, parse_dates=["date"])
    summary, daily = rolling_backtest(data, days=args.days)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "adaptive_backtest_summary.csv"
    daily_path = RESULTS_DIR / "adaptive_backtest_daily.csv"
    summary.to_csv(summary_path, index=False, date_format="%Y-%m-%d")
    daily.to_csv(daily_path, index=False, date_format="%Y-%m-%d")
    print(summary.to_string(index=False))
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved daily: {daily_path}")


if __name__ == "__main__":
    main()
