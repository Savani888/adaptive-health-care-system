"""Tests for classical allocation solvers and resource constraints."""

import numpy as np
import pytest

from config import DEPARTMENTS, NURSE_PATIENT_RATIOS
from allocation import (
    _check_constraints,
    compare_allocations,
)


@pytest.mark.parametrize("instance_seed", range(50))
def test_dp_and_ilp_match_on_random_instances(instance_seed: int) -> None:
    rng = np.random.default_rng(2026 + instance_seed)
    demand = {
        department: int(rng.integers(20, 151))
        for department in DEPARTMENTS
    }
    total_beds = int(rng.integers(40, 321))
    total_nurses = int(rng.integers(15, 81))
    ilp, dp = compare_allocations(demand, total_beds, total_nurses)
    assert ilp.objective == pytest.approx(dp.objective)
    assert ilp.bed_usage <= total_beds
    assert dp.bed_usage <= total_beds
    assert ilp.nurse_usage <= total_nurses
    assert dp.nurse_usage <= total_nurses
    for department in DEPARTMENTS:
        assert ilp.patients_served[department] <= demand[department]
        assert dp.patients_served[department] <= demand[department]


def test_constraints_are_always_checked() -> None:
    demand = {department: 100 for department in DEPARTMENTS}
    patients = {department: 10 for department in DEPARTMENTS}
    nurses = {
        department: 10 for department in DEPARTMENTS
    }
    _check_constraints(demand, patients, nurses, 40, 40)

    with pytest.raises(RuntimeError):
        _check_constraints(
            demand,
            patients,
            {department: 0 for department in DEPARTMENTS},
            40,
            40,
        )
