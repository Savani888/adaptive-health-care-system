"""Classical bed and nurse allocation using PuLP and exact dynamic programming."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pulp

from config import (
    DEPARTMENTS,
    DEFAULT_TOTAL_BEDS,
    DEFAULT_TOTAL_NURSES,
    NURSE_PATIENT_RATIOS,
    RESULTS_DIR,
    URGENCY_WEIGHTS,
)


@dataclass(frozen=True)
class AllocationResult:
    """One allocation, including both raw service and resource assignments."""

    patients_served: dict[str, int]
    nurse_allocations: dict[str, int]
    objective: float
    method: str
    bed_usage: int
    nurse_usage: int


def validate_inputs(
    forecast_demand: dict[str, float | int],
    total_beds: int,
    total_nurses: int,
) -> dict[str, int]:
    """Normalize demand to integer patient counts and reject invalid inputs."""
    if total_beds < 0 or total_nurses < 0:
        raise ValueError("Bed and nurse capacities must be non-negative.")
    demand: dict[str, int] = {}
    for department in DEPARTMENTS:
        if department not in forecast_demand:
            raise ValueError(f"Missing forecast demand for {department}.")
        value = float(forecast_demand[department])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Demand for {department} must be finite and non-negative.")
        demand[department] = int(np.floor(value))
    return demand


def solve_allocation_ilp(
    forecast_demand: dict[str, float | int],
    total_beds: int,
    total_nurses: int,
) -> AllocationResult:
    """Solve the integer allocation problem with PuLP's CBC solver."""
    demand = validate_inputs(forecast_demand, total_beds, total_nurses)
    model = pulp.LpProblem("bed_nurse_allocation", pulp.LpMaximize)
    served = {
        department: pulp.LpVariable(
            f"served_{department.replace(' ', '_')}",
            lowBound=0,
            upBound=demand[department],
            cat=pulp.LpInteger,
        )
        for department in DEPARTMENTS
    }

    model += pulp.lpSum(
        URGENCY_WEIGHTS[department] * served[department]
        for department in DEPARTMENTS
    )
    model += pulp.lpSum(served.values()) <= total_beds

    # The nurse constraint requires ceil(s/r_d), which linearization in PuLP
    # is not directly supported. Add an integer nurse variable per department
    # and enforce r_d*nurses >= s while summing those variables.
    nurse_variables: dict[str, pulp.LpVariable] = {}
    for department in DEPARTMENTS:
        nurses = pulp.LpVariable(
            f"nurses_{department.replace(' ', '_')}",
            lowBound=0,
            cat=pulp.LpInteger,
        )
        nurse_variables[department] = nurses
        ratio = NURSE_PATIENT_RATIOS[department]
        model += nurses * ratio >= served[department]
    model += pulp.lpSum(nurse_variables.values()) <= total_nurses
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[model.status] != "Optimal":
        raise RuntimeError(
            f"PuLP allocation did not reach an optimum: {pulp.LpStatus[model.status]}."
        )

    patients = {
        department: int(round(pulp.value(served[department]) or 0))
        for department in DEPARTMENTS
    }
    nurses = {
        department: int(round(pulp.value(nurse_variables[department]) or 0))
        for department in DEPARTMENTS
    }
    objective = float(pulp.value(model.objective) or 0.0)
    return _add_leftovers(
        demand=demand,
        patients=patients,
        nurse_allocations=nurses,
        total_beds=total_beds,
        total_nurses=total_nurses,
        objective=objective,
        method="PuLP CBC",
    )


def solve_allocation_dp(
    forecast_demand: dict[str, float | int],
    total_beds: int,
    total_nurses: int,
) -> AllocationResult:
    """Find the exact optimum with three-level dynamic programming."""
    demand = validate_inputs(forecast_demand, total_beds, total_nurses)

    # dp[bed_count][nurse_count] is the best weighted service for departments
    # processed so far. This is exact because bed and nurse totals are integers.
    dp = np.full((total_beds + 1, total_nurses + 1), -np.inf)
    dp[0, 0] = 0.0
    for department in DEPARTMENTS:
        next_dp = np.full((total_beds + 1, total_nurses + 1), -np.inf)
        ratio = NURSE_PATIENT_RATIOS[department]
        weight = URGENCY_WEIGHTS[department]
        for beds_used in range(total_beds + 1):
            for nurses_used in range(total_nurses + 1):
                if not np.isfinite(dp[beds_used, nurses_used]):
                    continue
                for patients in range(demand[department] + 1):
                    nurses = ceil(patients / ratio)
                    new_beds = beds_used + patients
                    new_nurses = nurses_used + nurses
                    if new_beds > total_beds or new_nurses > total_nurses:
                        continue
                    value = dp[beds_used, nurses_used] + weight * patients
                    if value > next_dp[new_beds, new_nurses]:
                        next_dp[new_beds, new_nurses] = value
        dp = next_dp

    objective = float(np.max(dp))
    if not np.isfinite(objective):
        raise RuntimeError("Dynamic programming found no feasible allocation.")
    _, nurse_allocations, patients = _reconstruct_dp(
        demand=demand,
        total_beds=total_beds,
        total_nurses=total_nurses,
        objective=objective,
    )
    return _add_leftovers(
        demand=demand,
        patients=patients,
        nurse_allocations=nurse_allocations,
        total_beds=total_beds,
        total_nurses=total_nurses,
        objective=objective,
        method="Exact DP",
    )


def _reconstruct_dp(
    demand: dict[str, int],
    total_beds: int,
    total_nurses: int,
    objective: float,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Recover the service vector associated with the DP optimum."""
    dp = np.full((total_beds + 1, total_nurses + 1), -np.inf)
    dp[0, 0] = 0.0
    for department in DEPARTMENTS:
        next_dp = np.full((total_beds + 1, total_nurses + 1), -np.inf)
        ratio = NURSE_PATIENT_RATIOS[department]
        weight = URGENCY_WEIGHTS[department]
        for beds_used in range(total_beds + 1):
            for nurses_used in range(total_nurses + 1):
                if not np.isfinite(dp[beds_used, nurses_used]):
                    continue
                for patients in range(demand[department] + 1):
                    new_beds = beds_used + patients
                    new_nurses = nurses_used + ceil(patients / ratio)
                    if new_beds <= total_beds and new_nurses <= total_nurses:
                        value = dp[beds_used, nurses_used] + weight * patients
                        next_dp[new_beds, new_nurses] = max(
                            next_dp[new_beds, new_nurses],
                            value,
                        )
        dp = next_dp

    best_beds = int(np.argwhere(dp == objective)[-1, 0])
    best_nurses = int(np.argwhere(dp == objective)[-1, 1])

    # Store only the first exact state recurrence needed for reconstruction.
    states = [(0, 0)]
    for department in DEPARTMENTS:
        # This is intentionally replaced below by a forward state table.
        states.append(states[-1])

    dp_with_choices: list[np.ndarray] = [
        np.full((total_beds + 1, total_nurses + 1), -np.inf)
    ]
    dp_with_choices[0][0, 0] = 0.0
    choices: list[dict[tuple[int, int], tuple[int, int]]] = [{}]

    for department in DEPARTMENTS:
        current = dp_with_choices[-1].copy()
        next_dp = np.full((total_beds + 1, total_nurses + 1), -np.inf)
        step_choices: dict[tuple[int, int], tuple[int, int]] = {}
        ratio = NURSE_PATIENT_RATIOS[department]
        weight = URGENCY_WEIGHTS[department]
        for beds_used in range(total_beds + 1):
            for nurses_used in range(total_nurses + 1):
                if not np.isfinite(current[beds_used, nurses_used]):
                    continue
                for patients in range(demand[department] + 1):
                    new_beds = beds_used + patients
                    new_nurses = nurses_used + ceil(patients / ratio)
                    if new_beds > total_beds or new_nurses > total_nurses:
                        continue
                    value = current[beds_used, nurses_used] + weight * patients
                    if value > next_dp[new_beds, new_nurses]:
                        next_dp[new_beds, new_nurses] = value
                        step_choices[(new_beds, new_nurses)] = (
                            beds_used,
                            nurses_used,
                            patients,
                        )
        dp_with_choices.append(next_dp)
        choices.append(step_choices)

    beds_now, nurses_now = best_beds, best_nurses
    patient_values = {department: 0 for department in DEPARTMENTS}
    nurse_values = {department: 0 for department in DEPARTMENTS}
    for department_index in range(len(DEPARTMENTS) - 1, -1, -1):
        department = DEPARTMENTS[department_index]
        previous_beds, previous_nurses, patients = choices[department_index + 1][
            (beds_now, nurses_now)
        ]
        patient_values[department] = patients
        nurse_values[department] = ceil(
            patients / NURSE_PATIENT_RATIOS[department]
        )
        beds_now, nurses_now = previous_beds, previous_nurses
    return nurse_values, nurse_values, patient_values


def _add_leftovers(
    demand: dict[str, int],
    patients: dict[str, int],
    nurse_allocations: dict[str, int],
    total_beds: int,
    total_nurses: int,
    objective: float,
    method: str,
) -> AllocationResult:
    """Use remaining beds and nurses in proportion to unmet and demand."""
    patients = dict(patients)
    nurse_allocations = dict(nurse_allocations)

    bed_usage = sum(patients.values())
    leftover_beds = max(0, total_beds - bed_usage)
    nurse_usage = sum(nurse_allocations.values())
    leftover_nurses = max(0, total_nurses - nurse_usage)

    # Assign leftover nurses using largest-remainder proportional apportionment.
    if leftover_nurses:
        nurse_demands = {
            department: max(0, demand[department] - patients[department])
            for department in DEPARTMENTS
        }
        nurse_weights = _largest_remainder_apportionment(
            nurse_demands,
            leftover_nurses,
        )
        for department in DEPARTMENTS:
            nurse_allocations[department] += nurse_weights[department]

    # Allocate leftover beds only if the department has enough nurse capacity.
    for _ in range(leftover_beds):
        eligible = [
            department
            for department in DEPARTMENTS
            if patients[department] < demand[department]
            and ceil((patients[department] + 1) / NURSE_PATIENT_RATIOS[department])
            <= nurse_allocations[department]
        ]
        if not eligible:
            break
        weights = np.array(
            [max(1, demand[department] - patients[department]) for department in eligible],
            dtype=float,
        )
        # Choose the department with the largest unmet-share, using stable
        # department order as a deterministic tie-breaker.
        chosen = eligible[int(np.argmax(weights))]
        patients[chosen] += 1

    bed_usage = sum(patients.values())
    objective = sum(
        URGENCY_WEIGHTS[department] * patients[department]
        for department in DEPARTMENTS
    )
    nurse_usage = sum(nurse_allocations.values())
    _check_constraints(
        demand=demand,
        patients=patients,
        nurse_allocations=nurse_allocations,
        total_beds=total_beds,
        total_nurses=total_nurses,
    )
    return AllocationResult(
        patients_served=patients,
        nurse_allocations=nurse_allocations,
        objective=objective,
        method=method,
        bed_usage=bed_usage,
        nurse_usage=nurse_usage,
    )


def _largest_remainder_apportionment(
    weights: dict[str, int],
    units: int,
) -> dict[str, int]:
    """Distribute units proportionally to weights using largest remainders."""
    total_weight = sum(weights.values())
    result = {department: 0 for department in DEPARTMENTS}
    if total_weight == 0 or units == 0:
        return result
    exact = {
        department: units * weights[department] / total_weight
        for department in DEPARTMENTS
    }
    for department in DEPARTMENTS:
        result[department] = int(np.floor(exact[department]))
    remaining = units - sum(result.values())
    ranked = sorted(
        DEPARTMENTS,
        key=lambda department: (
            -(exact[department] - result[department]),
            DEPARTMENTS.index(department),
        ),
    )
    for department in ranked[:remaining]:
        result[department] += 1
    return result


def _check_constraints(
    demand: dict[str, int],
    patients: dict[str, int],
    nurse_allocations: dict[str, int],
    total_beds: int,
    total_nurses: int,
) -> None:
    """Raise on any feasibility violation."""
    if sum(patients.values()) > total_beds:
        raise RuntimeError("Allocation exceeds bed capacity.")
    if sum(nurse_allocations.values()) > total_nurses:
        raise RuntimeError("Allocation exceeds nurse capacity.")
    for department in DEPARTMENTS:
        if patients[department] > demand[department]:
            raise RuntimeError("Allocation exceeds forecast demand.")
        required = ceil(patients[department] / NURSE_PATIENT_RATIOS[department])
        if nurse_allocations[department] < required:
            raise RuntimeError("Department lacks enough assigned nurses.")


def compare_allocations(
    forecast_demand: dict[str, float | int],
    total_beds: int,
    total_nurses: int,
) -> tuple[AllocationResult, AllocationResult]:
    """Run both classical solvers on the same instance."""
    return (
        solve_allocation_ilp(forecast_demand, total_beds, total_nurses),
        solve_allocation_dp(forecast_demand, total_beds, total_nurses),
    )


def main() -> None:
    """Allocate the first forecast day's demand and save a real result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecast",
        type=Path,
        default=RESULTS_DIR / "seven_day_forecast.csv",
    )
    parser.add_argument("--beds", type=int, default=DEFAULT_TOTAL_BEDS)
    parser.add_argument("--nurses", type=int, default=DEFAULT_TOTAL_NURSES)
    args = parser.parse_args()

    forecast = pd.read_csv(args.forecast)
    first_date = forecast["date"].min()
    demand: dict[str, float] = {}
    for department in DEPARTMENTS:
        row = forecast[
            (forecast["date"] == first_date)
            & (forecast["department"] == department)
        ]
        demand[department] = float(row["forecast_census"].iloc[0])

    ilp, dp = compare_allocations(demand, args.beds, args.nurses)
    records = []
    for result in (ilp, dp):
        record: dict[str, object] = {
            "method": result.method,
            "objective": result.objective,
            "bed_usage": result.bed_usage,
            "nurse_usage": result.nurse_usage,
            "data_type": "synthetic",
        }
        record.update(
            {
                f"{department}_patients": result.patients_served[department]
                for department in DEPARTMENTS
            }
        )
        record.update(
            {
                f"{department}_nurses": result.nurse_allocations[department]
                for department in DEPARTMENTS
            }
        )
        records.append(record)

    results = pd.DataFrame(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "allocation_comparison.csv"
    results.to_csv(output_path, index=False)
    print(results.to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
