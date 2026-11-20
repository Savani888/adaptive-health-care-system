"""Tests for forecasting outputs and target-date leakage protection."""

import numpy as np
import pandas as pd
import pytest

from config import DEPARTMENTS, FORECAST_HORIZONS_DAYS
from data_gen import generate_synthetic_data
from forecasting import (
    build_supervised_frame,
    evaluate_forecasts,
    fit_forecasters,
    forecast_next_7_days,
)


@pytest.fixture(scope="module")
def synthetic_data() -> pd.DataFrame:
    return generate_synthetic_data(output_path=None)


def test_training_labels_do_not_cross_the_cutoff(
    synthetic_data: pd.DataFrame,
) -> None:
    cutoff = pd.Timestamp("2025-10-01")
    bundle = fit_forecasters(synthetic_data, train_until=cutoff)
    assert len(bundle.models) == len(DEPARTMENTS) * len(FORECAST_HORIZONS_DAYS)
    assert all(date <= cutoff for date in bundle.training_target_max.values())
    assert all(count >= 60 for count in bundle.training_rows.values())


def test_future_observations_cannot_change_a_past_origin_forecast(
    synthetic_data: pd.DataFrame,
) -> None:
    department = DEPARTMENTS[0]
    small_history = synthetic_data.loc[
        (synthetic_data["department"] == department)
        & (synthetic_data["date"] <= pd.Timestamp("2023-08-31"))
    ].copy()
    as_of = pd.Timestamp("2023-07-31")
    changed_future = small_history.copy()
    future_rows = changed_future["date"] > as_of
    changed_future.loc[future_rows, ["admissions", "census"]] = 1_000_000

    first = forecast_next_7_days(small_history, as_of=as_of)
    second = forecast_next_7_days(changed_future, as_of=as_of)
    np.testing.assert_allclose(
        first["forecast_census"].to_numpy(),
        second["forecast_census"].to_numpy(),
    )


def test_supervised_features_end_at_origin_and_targets_match_horizon(
    synthetic_data: pd.DataFrame,
) -> None:
    department = DEPARTMENTS[0]
    horizon = 7
    examples = build_supervised_frame(synthetic_data, department, horizon)
    sample = examples.iloc[-20]
    assert sample["target_date"] - sample["origin_date"] == pd.Timedelta(days=horizon)
    assert sample["target_census"] == synthetic_data.loc[
        (synthetic_data["department"] == department)
        & (synthetic_data["date"] == sample["target_date"]),
        "census",
    ].iloc[0]
    assert "census_lag_0" in examples
    assert "admissions_mean_28" in examples


def test_metrics_cover_each_department_and_horizon(
    synthetic_data: pd.DataFrame,
) -> None:
    metrics = evaluate_forecasts(synthetic_data, test_days=28)
    assert len(metrics) == len(DEPARTMENTS) * len(FORECAST_HORIZONS_DAYS)
    assert metrics["data_type"].eq("synthetic").all()
    assert (metrics["model_mae"] >= 0).all()
    assert (metrics["model_rmse"] >= 0).all()
    assert (metrics["model_mape_pct"] >= 0).all()
