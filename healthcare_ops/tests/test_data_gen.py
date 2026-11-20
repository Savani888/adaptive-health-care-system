"""Focused tests for module 1: synthetic data generation."""

import pandas as pd

from data_gen import generate_synthetic_data


def test_generation_is_deterministic() -> None:
    first = generate_synthetic_data(output_path=None)
    second = generate_synthetic_data(output_path=None)
    pd.testing.assert_frame_equal(first, second)


def test_data_has_expected_shape_and_synthetic_label() -> None:
    data = generate_synthetic_data(output_path=None)
    assert len(data) == (3 * 365 + 1) * 4  # One row per day and department.
    assert data.groupby("department")["date"].nunique().eq(1096).all()
    assert data["department"].nunique() == 4
    assert set(data["data_type"]) == {"synthetic"}
    assert (data["admissions"] >= 0).all()
    assert (data["census"] >= data["admissions"]).all()
    assert data["outbreak"].any()
