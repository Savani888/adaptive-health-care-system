"""Tests for rolling adaptive backtest structure and strategy outputs."""

import pandas as pd

from adaptive import STRATEGIES, rolling_backtest
from data_gen import generate_synthetic_data


def test_backtest_returns_all_seven_strategies_and_periods() -> None:
    data = generate_synthetic_data(output_path=None)
    summary, daily = rolling_backtest(data, days=56)

    assert set(summary["strategy"]) == set(STRATEGIES)
    assert set(summary["period"]).issubset({"normal", "outbreak"})
    assert len(daily) == 56 * len(STRATEGIES)
    assert daily["data_type"].eq("synthetic").all()
    assert (daily["weighted_service_level"] >= 0).all()
    assert (daily["weighted_service_level"] <= 1).all()
    assert (daily["unmet_patients"] >= 0).all()
    assert (daily["unmet_emergency_patients"] >= 0).all()
