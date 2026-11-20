"""Leakage-safe census forecasting and baseline evaluation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from config import (
    DEPARTMENTS,
    DEFAULT_DATA_PATH,
    FORECAST_HORIZONS_DAYS,
    FORECAST_LAGS_DAYS,
    FORECAST_LEARNING_RATE,
    FORECAST_LOSS,
    FORECAST_MAX_DEPTH,
    FORECAST_MIN_SAMPLES_LEAF,
    FORECAST_MIN_TRAINING_ROWS,
    FORECAST_N_ESTIMATORS,
    FORECAST_ROLLING_WINDOWS_DAYS,
    FORECAST_TEST_DAYS,
    RANDOM_SEED,
    RESULTS_DIR,
)

REQUIRED_COLUMNS = {"date", "department", "admissions", "census"}
FEATURE_COLUMNS = (
    tuple(f"census_lag_{lag}" for lag in FORECAST_LAGS_DAYS)
    + tuple(f"admissions_lag_{lag}" for lag in FORECAST_LAGS_DAYS)
    + tuple(
        f"{measure}_mean_{window}"
        for window in FORECAST_ROLLING_WINDOWS_DAYS
        for measure in ("census", "admissions")
    )
    + (
        "target_weekday",
        "target_month",
        "target_is_weekend",
        "target_day_of_year_sin",
        "target_day_of_year_cos",
    )
)


@dataclass
class ForecastBundle:
    """Models and the date boundary used to prevent target leakage."""

    models: dict[tuple[str, int], GradientBoostingRegressor]
    training_cutoff: pd.Timestamp
    training_target_max: dict[tuple[str, int], pd.Timestamp]
    training_rows: dict[tuple[str, int], int]


def _validated_data(data: pd.DataFrame) -> pd.DataFrame:
    """Check the input schema, dates, numeric values, and unique daily keys."""
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Forecast input is missing columns: {sorted(missing)}")

    result = data.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    if result[["department", "date"]].duplicated().any():
        raise ValueError("Forecast input must have one row per department and date.")
    for column in ("admissions", "census"):
        result[column] = pd.to_numeric(result[column], errors="raise")
        if result[column].isna().any() or (result[column] < 0).any():
            raise ValueError(f"{column} must contain non-negative numeric values.")
    return result.sort_values(["department", "date"]).reset_index(drop=True)


def _department_data(data: pd.DataFrame, department: str) -> pd.DataFrame:
    """Return one department's daily history and reject missing calendar days."""
    history = data.loc[data["department"] == department].sort_values("date")
    if history.empty:
        raise ValueError(f"No observations found for department {department!r}.")
    history = history.reset_index(drop=True)
    if len(history) > 1:
        gaps = history["date"].diff().dropna()
        if not gaps.eq(pd.Timedelta(days=1)).all():
            raise ValueError(
                f"{department} history must have exactly one observation per day."
            )
    return history


def _feature_values(
    history: pd.DataFrame,
    origin_index: int,
    target_date: pd.Timestamp,
) -> dict[str, float]:
    """Build features using observations through origin_index only."""
    if origin_index < max(
        max(FORECAST_LAGS_DAYS), max(FORECAST_ROLLING_WINDOWS_DAYS) - 1
    ):
        raise ValueError("At least 28 days of history are required for forecasting.")

    features: dict[str, float] = {}
    for measure in ("census", "admissions"):
        values = history[measure]
        for lag in FORECAST_LAGS_DAYS:
            features[f"{measure}_lag_{lag}"] = float(values.iloc[origin_index - lag])
        for window in FORECAST_ROLLING_WINDOWS_DAYS:
            start = origin_index - window + 1
            features[f"{measure}_mean_{window}"] = float(
                values.iloc[start : origin_index + 1].mean()
            )

    target_date = pd.Timestamp(target_date)
    day_of_year = target_date.dayofyear - 1
    yearly_angle = 2.0 * np.pi * day_of_year / 365.25
    features.update(
        {
            "target_weekday": float(target_date.weekday()),
            "target_month": float(target_date.month),
            "target_is_weekend": float(target_date.weekday() >= 5),
            "target_day_of_year_sin": float(np.sin(yearly_angle)),
            "target_day_of_year_cos": float(np.cos(yearly_angle)),
        }
    )
    return features


def build_supervised_frame(
    data: pd.DataFrame,
    department: str,
    horizon: int,
) -> pd.DataFrame:
    """Create examples whose features end on the origin and label is future census.

    For a row with origin date t and horizon h, the label is census at t+h.
    No observation after t is used to calculate the row's features.
    """
    if horizon not in FORECAST_HORIZONS_DAYS:
        raise ValueError(f"horizon must be one of {FORECAST_HORIZONS_DAYS}.")
    validated = _validated_data(data)
    history = _department_data(validated, department)
    return _supervised_frame_from_history(history, horizon)


def _supervised_frame_from_history(
    history: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Vectorize example creation for a validated, continuous daily series."""
    min_origin = max(
        max(FORECAST_LAGS_DAYS), max(FORECAST_ROLLING_WINDOWS_DAYS) - 1
    )
    origins = np.arange(min_origin, len(history) - horizon, dtype=int)
    targets = origins + horizon
    dates = history["date"].reset_index(drop=True)
    target_dates = dates.iloc[targets].reset_index(drop=True)

    values: dict[str, object] = {
        "origin_date": dates.iloc[origins].to_numpy(),
        "target_date": target_dates.to_numpy(),
    }
    for measure in ("census", "admissions"):
        measure_values = history[measure].to_numpy(dtype=float)
        for lag in FORECAST_LAGS_DAYS:
            values[f"{measure}_lag_{lag}"] = measure_values[origins - lag]
        for window in FORECAST_ROLLING_WINDOWS_DAYS:
            rolling_means = (
                pd.Series(measure_values).rolling(window=window).mean().to_numpy()
            )
            values[f"{measure}_mean_{window}"] = rolling_means[origins]

    weekdays = target_dates.dt.weekday.to_numpy(dtype=float)
    months = target_dates.dt.month.to_numpy(dtype=float)
    angles = 2.0 * np.pi * (target_dates.dt.dayofyear.to_numpy() - 1) / 365.25
    values.update(
        {
            "target_weekday": weekdays,
            "target_month": months,
            "target_is_weekend": (weekdays >= 5).astype(float),
            "target_day_of_year_sin": np.sin(angles),
            "target_day_of_year_cos": np.cos(angles),
            "target_census": history["census"].to_numpy(dtype=float)[targets],
        }
    )
    return pd.DataFrame(values).loc[
        :, ["origin_date", "target_date", *FEATURE_COLUMNS, "target_census"]
    ]


def fit_forecasters(
    data: pd.DataFrame,
    train_until: str | pd.Timestamp | None = None,
    horizons: tuple[int, ...] = FORECAST_HORIZONS_DAYS,
) -> ForecastBundle:
    """Fit one GradientBoostingRegressor per available department and horizon.

    Samples are selected by their label's target date, never by their origin
    date. Consequently, no future target enters model fitting.
    """
    validated = _validated_data(data)
    cutoff = (
        pd.Timestamp(train_until).normalize()
        if train_until is not None
        else validated["date"].max()
    )
    models: dict[tuple[str, int], GradientBoostingRegressor] = {}
    target_max: dict[tuple[str, int], pd.Timestamp] = {}
    row_counts: dict[tuple[str, int], int] = {}

    available_departments = set(validated["department"])
    for department in DEPARTMENTS:
        if department not in available_departments:
            continue
        history = _department_data(validated, department)
        for horizon in horizons:
            examples = _supervised_frame_from_history(history, horizon)
            train = examples.loc[examples["target_date"] <= cutoff]
            if len(train) < FORECAST_MIN_TRAINING_ROWS:
                raise ValueError(
                    f"Insufficient training examples for {department}, "
                    f"horizon {horizon}: found {len(train)}, need at least "
                    f"{FORECAST_MIN_TRAINING_ROWS}."
                )

            model = GradientBoostingRegressor(
                n_estimators=FORECAST_N_ESTIMATORS,
                learning_rate=FORECAST_LEARNING_RATE,
                max_depth=FORECAST_MAX_DEPTH,
                min_samples_leaf=FORECAST_MIN_SAMPLES_LEAF,
                loss=FORECAST_LOSS,
                random_state=RANDOM_SEED,
            )
            model.fit(train.loc[:, FEATURE_COLUMNS], train["target_census"])
            key = (department, horizon)
            models[key] = model
            target_max[key] = pd.Timestamp(train["target_date"].max())
            row_counts[key] = len(train)

    if not models:
        raise ValueError("No department models could be fitted.")
    return ForecastBundle(models, cutoff, target_max, row_counts)


def forecast_next_day(
    data: pd.DataFrame,
    as_of: str | pd.Timestamp,
    bundle: ForecastBundle,
) -> pd.DataFrame:
    """Forecast horizon-1 census only for the rolling backtest."""
    validated = _validated_data(data)
    origin = pd.Timestamp(as_of).normalize()
    output: list[dict[str, object]] = []
    target = origin + pd.Timedelta(days=1)
    for department in DEPARTMENTS:
        history = _department_data(validated, department)
        known = history.loc[history["date"] <= origin].reset_index(drop=True)
        matching = known.index[known["date"] == origin]
        if len(matching) != 1:
            raise ValueError(
                f"Forecast origin {origin.date()} is not present for {department}."
            )
        origin_index = int(matching[0])
        features = _feature_values(known, origin_index, target)
        key = (department, 1)
        if key not in bundle.models:
            raise ValueError(f"No horizon-1 model for {department}.")
        predicted = float(
            bundle.models[key].predict(
                pd.DataFrame([features], columns=FEATURE_COLUMNS)
            )[0]
        )
        output.append({
            "date": target,
            "department": department,
            "horizon": 1,
            "forecast_census": max(0.0, predicted),
            "data_type": "synthetic",
        })
    return pd.DataFrame(output)


def forecast_next_7_days(
    data: pd.DataFrame,
    as_of: str | pd.Timestamp | None = None,
    bundle: ForecastBundle | None = None,
) -> pd.DataFrame:
    """Forecast each available department's census for the next seven days."""
    validated = _validated_data(data)
    forecast_date = (
        pd.Timestamp(as_of).normalize() if as_of is not None else validated["date"].max()
    )
    if bundle is None:
        bundle = fit_forecasters(validated, train_until=forecast_date)
    elif bundle.training_cutoff > forecast_date:
        raise ValueError("Model training cutoff cannot be after the forecast origin.")

    output: list[dict[str, object]] = []
    for department in DEPARTMENTS:
        if department not in set(validated["department"]):
            continue
        history = _department_data(validated, department)
        known_history = history.loc[history["date"] <= forecast_date].reset_index(
            drop=True
        )
        matching = known_history.index[known_history["date"] == forecast_date]
        if len(matching) != 1:
            raise ValueError(
                f"Forecast origin {forecast_date.date()} is not present for {department}."
            )
        origin_index = int(matching[0])
        prior_census = known_history.set_index("date")["census"]

        for horizon in FORECAST_HORIZONS_DAYS:
            key = (department, horizon)
            if key not in bundle.models:
                raise ValueError(f"No fitted model for {department}, horizon {horizon}.")
            target_date = forecast_date + pd.Timedelta(days=horizon)
            features = _feature_values(known_history, origin_index, target_date)
            prediction = float(
                bundle.models[key].predict(
                    pd.DataFrame([features], columns=FEATURE_COLUMNS)
                )[0]
            )
            last_week_date = target_date - pd.Timedelta(days=7)
            if last_week_date not in prior_census.index:
                raise ValueError("Insufficient history for the same-weekday baseline.")
            output.append(
                {
                    "date": target_date,
                    "department": department,
                    "horizon": horizon,
                    "forecast_census": max(0.0, prediction),
                    "baseline_last_week": float(prior_census.loc[last_week_date]),
                    "baseline_28day_mean": features["census_mean_28"],
                    "data_type": "synthetic",
                }
            )
    return pd.DataFrame(output)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    """Calculate MAE, RMSE, and zero-safe MAPE."""
    errors = predicted - actual
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    mape = float(np.mean(np.abs(errors) / np.maximum(np.abs(actual), 1.0)) * 100.0)
    return mae, rmse, mape


def evaluate_forecasts(
    data: pd.DataFrame,
    test_days: int = FORECAST_TEST_DAYS,
) -> pd.DataFrame:
    """Evaluate fixed training models and two baselines on a chronological holdout.

    Models are trained only with labels dated on or before the holdout cutoff.
    During the holdout, features may use observations available at each origin,
    simulating a daily rolling forecast without retraining the fitted models.
    """
    if test_days < 1:
        raise ValueError("test_days must be at least one.")
    validated = _validated_data(data)
    latest_date = validated["date"].max()
    cutoff = latest_date - pd.Timedelta(days=test_days)
    if cutoff <= validated["date"].min():
        raise ValueError("Not enough history before the requested test window.")
    bundle = fit_forecasters(validated, train_until=cutoff)
    reports: list[dict[str, object]] = []

    for department in DEPARTMENTS:
        if department not in set(validated["department"]):
            continue
        history = _department_data(validated, department)
        census_by_date = history.set_index("date")["census"]
        for horizon in FORECAST_HORIZONS_DAYS:
            key = (department, horizon)
            examples = _supervised_frame_from_history(history, horizon)
            test = examples.loc[
                (examples["target_date"] > cutoff)
                & (examples["target_date"] <= latest_date)
            ]
            if test.empty:
                raise ValueError(
                    f"No test examples for {department}, horizon {horizon}."
                )

            actual = test["target_census"].to_numpy(dtype=float)
            model_values = np.maximum(
                bundle.models[key].predict(test.loc[:, FEATURE_COLUMNS]), 0.0
            )
            last_week_dates = test["target_date"] - pd.Timedelta(days=7)
            if not last_week_dates.isin(census_by_date.index).all():
                raise ValueError("Insufficient history for the same-weekday baseline.")
            last_week_values = census_by_date.reindex(last_week_dates).to_numpy(
                dtype=float
            )
            average_values = test["census_mean_28"].to_numpy(dtype=float)

            model_mae, model_rmse, model_mape = _metrics(actual, model_values)
            week_mae, week_rmse, week_mape = _metrics(actual, last_week_values)
            mean_mae, mean_rmse, mean_mape = _metrics(actual, average_values)
            reports.append(
                {
                    "department": department,
                    "horizon": horizon,
                    "test_rows": len(test),
                    "train_cutoff": cutoff,
                    "latest_training_target": bundle.training_target_max[key],
                    "model_mae": model_mae,
                    "model_rmse": model_rmse,
                    "model_mape_pct": model_mape,
                    "last_week_mae": week_mae,
                    "last_week_rmse": week_rmse,
                    "last_week_mape_pct": week_mape,
                    "mean_28day_mae": mean_mae,
                    "mean_28day_rmse": mean_rmse,
                    "mean_28day_mape_pct": mean_mape,
                    "data_type": "synthetic",
                }
            )

    return pd.DataFrame(reports).sort_values(["department", "horizon"]).reset_index(
        drop=True
    )


def main() -> None:
    """Generate the next-week outlook and chronological holdout metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    data = pd.read_csv(args.data, parse_dates=["date"])
    metrics = evaluate_forecasts(data)
    forecasts = forecast_next_7_days(data)
    args.results.mkdir(parents=True, exist_ok=True)
    metrics_path = args.results / "forecast_metrics.csv"
    forecasts_path = args.results / "seven_day_forecast.csv"
    metrics.to_csv(metrics_path, index=False, date_format="%Y-%m-%d")
    forecasts.to_csv(forecasts_path, index=False, date_format="%Y-%m-%d")

    print("Synthetic holdout metrics (averaged MAE by forecast approach):")
    print(
        metrics[
            ["model_mae", "last_week_mae", "mean_28day_mae"]
        ].mean().round(2).to_string()
    )
    print(f"\nNext 7 days from {data['date'].max().date()} (synthetic):")
    print(forecasts.head(8).to_string(index=False))
    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved outlook: {forecasts_path}")


if __name__ == "__main__":
    main()
