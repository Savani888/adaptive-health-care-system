"""Generate reproducible synthetic daily admissions and patient census."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BASE_DAILY_ADMISSIONS,
    DATA_END,
    DATA_START,
    DEPARTMENTS,
    DEFAULT_DATA_PATH,
    MEAN_LENGTH_OF_STAY_DAYS,
    OUTBREAK_DEPARTMENT_MULTIPLIERS,
    OUTBREAK_WAVES,
    RANDOM_SEED,
    WINTER_ADMISSION_MULTIPLIER,
    WINTER_MONTHS,
    WEEKDAY_ADMISSION_MULTIPLIERS,
    YEARLY_ADMISSION_GROWTH,
)


def _outbreak_multiplier(date: pd.Timestamp, department: str) -> float:
    """Return the combined effect of any outbreak active on this date."""
    multiplier = 1.0
    for wave in OUTBREAK_WAVES:
        if pd.Timestamp(wave["start"]) <= date <= pd.Timestamp(wave["end"]):
            multiplier *= wave["multiplier"] * OUTBREAK_DEPARTMENT_MULTIPLIERS[
                department
            ]
    return multiplier


def generate_synthetic_data(
    output_path: str | Path | None = DEFAULT_DATA_PATH,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Create three years of daily observations and optionally save them.

    Each patient's length of stay is simulated independently. A patient's
    admission contributes to census on the admission date and on following
    dates until that sampled stay ends, creating the expected census lag.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(DATA_START, DATA_END, freq="D")
    rows: list[dict[str, object]] = []

    for department in DEPARTMENTS:
        # Daily cohorts store each patient's remaining stay at midnight.
        active_stays: list[int] = []
        for date in dates:
            elapsed_years = (date - dates[0]).days / 365.25
            growth = (1.0 + YEARLY_ADMISSION_GROWTH) ** elapsed_years
            weekday_effect = WEEKDAY_ADMISSION_MULTIPLIERS[department][
                date.weekday()
            ]
            winter_effect = (
                WINTER_ADMISSION_MULTIPLIER[department]
                if date.month in WINTER_MONTHS
                else 1.0
            )
            outbreak_effect = _outbreak_multiplier(date, department)
            expected_admissions = (
                BASE_DAILY_ADMISSIONS[department]
                * growth
                * weekday_effect
                * winter_effect
                * outbreak_effect
            )
            admissions = int(rng.poisson(expected_admissions))

            # Decrease stay lengths before adding today's new patients.
            active_stays = [days_left - 1 for days_left in active_stays if days_left > 1]
            new_stays = rng.poisson(
                MEAN_LENGTH_OF_STAY_DAYS[department], size=admissions
            )
            active_stays.extend(max(1, int(stay)) for stay in new_stays)
            census = len(active_stays)

            rows.append(
                {
                    "date": date,
                    "department": department,
                    "admissions": admissions,
                    "census": census,
                    "outbreak": outbreak_effect > 1.0,
                    "data_type": "synthetic",
                }
            )

    data = pd.DataFrame(rows)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(destination, index=False, date_format="%Y-%m-%d")
    return data


def main() -> None:
    """Command-line entry point for generating the CSV."""
    parser = argparse.ArgumentParser(
        description="Generate the project's synthetic hospital operations data."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="CSV output path (default: healthcare_ops/data/synthetic_daily_data.csv)",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    data = generate_synthetic_data(args.output, seed=args.seed)
    print(f"Generated {len(data):,} synthetic rows: {args.output}")
    print(data.head(8).to_string(index=False))
    print(f"Date range: {data['date'].min().date()} to {data['date'].max().date()}")
    print(f"Departments: {', '.join(DEPARTMENTS)}")


if __name__ == "__main__":
    main()
