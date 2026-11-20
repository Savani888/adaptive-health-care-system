"""Pure synthetic real-time hospital simulation helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from math import ceil

from config import (
    DEPARTMENTS,
    NURSE_PATIENT_RATIOS,
    SIMULATION_ARRIVAL_PER_TICK,
    SIMULATION_DISCHARGE_PROBABILITY,
    SIMULATION_INITIAL_OCCUPANCY,
    SIMULATION_SEED,
    URGENCY_WEIGHTS,
)


@dataclass
class SimulationState:
    """All mutable values for one synthetic hospital game."""

    occupied: dict[str, int]
    waiting: dict[str, int]
    tick: int = 0
    score: int = 0
    admitted: int = 0
    discharged: int = 0
    waiting_total: int = 0
    rng: random.Random = field(
        default_factory=lambda: random.Random(SIMULATION_SEED)
    )
    history: list[dict[str, object]] = field(default_factory=list)


def initialize_simulation(seed: int = SIMULATION_SEED) -> SimulationState:
    """Create a fresh synthetic hospital state."""
    return SimulationState(
        occupied={
            department: int(SIMULATION_INITIAL_OCCUPANCY[department])
            for department in DEPARTMENTS
        },
        waiting={department: 0 for department in DEPARTMENTS},
        rng=random.Random(seed),
    )


def nurse_requirement(patients: int, department: str) -> int:
    """Calculate the synthetic nurse requirement for a department."""
    return ceil(patients / NURSE_PATIENT_RATIOS[department])


def simulate_tick(
    state: SimulationState,
    total_beds: int,
    total_nurses: int,
    surge_percent: int = 0,
) -> SimulationState:
    """Advance the synthetic hospital by one time step."""
    if total_beds < 0 or total_nurses < 0:
        raise ValueError("Beds and nurses cannot be negative.")
    surge = 1.0 + max(0, surge_percent) / 100.0

    # A user can lower capacity below the current occupancy. Balance it first,
    # using urgency order to protect the most important departments.
    department_order = sorted(
        DEPARTMENTS,
        key=lambda department: (-URGENCY_WEIGHTS[department], department),
    )
    if sum(state.occupied.values()) > total_beds:
        excess_beds = sum(state.occupied.values()) - total_beds
        for department in reversed(department_order):
            removable = min(excess_beds, state.occupied[department])
            state.occupied[department] -= removable
            state.discharged += removable
            excess_beds -= removable
            if excess_beds == 0:
                break

    for department in reversed(department_order):
        while (
            sum(
                nurse_requirement(state.occupied[name], name)
                for name in DEPARTMENTS
            )
            > total_nurses
            and state.occupied[department] > 0
        ):
            state.occupied[department] -= 1
            state.discharged += 1

    # Existing patients have a chance to leave and free beds/nurses.
    for department in DEPARTMENTS:
        if state.occupied[department] <= 0:
            continue
        expected_discharges = (
            state.occupied[department]
            * SIMULATION_DISCHARGE_PROBABILITY[department]
        )
        discharges = 0
        probability = min(1.0, expected_discharges / state.occupied[department])
        for _ in range(state.occupied[department]):
            if state.rng.random() < probability:
                discharges += 1
        discharges = min(discharges, state.occupied[department])
        state.occupied[department] -= discharges
        state.discharged += discharges

    # Each tick has Poisson-like synthetic arrivals, sampled from configured rates.
    for department in DEPARTMENTS:
        mean = SIMULATION_ARRIVAL_PER_TICK[department] * surge
        arrivals = poisson_sample(state.rng, mean)
        state.waiting[department] += arrivals

    state.tick += 1

    # Admit urgent departments first, but still cap all admissions by beds and nurses.
    total_beds_used = sum(state.occupied.values())
    total_nurses_used = sum(
        nurse_requirement(state.occupied[department], department)
        for department in DEPARTMENTS
    )
    department_order = sorted(
        DEPARTMENTS,
        key=lambda department: (-URGENCY_WEIGHTS[department], department),
    )
    for department in department_order:
        while state.waiting[department] > 0:
            if total_beds_used >= total_beds:
                break
            required_nurses = nurse_requirement(
                state.occupied[department] + 1,
                department,
            )
            increase = required_nurses - nurse_requirement(
                state.occupied[department],
                department,
            )
            if total_nurses_used + increase > total_nurses:
                break
            state.waiting[department] -= 1
            state.occupied[department] += 1
            total_beds_used += 1
            total_nurses_used += increase
            state.admitted += 1
            state.score += URGENCY_WEIGHTS[department]

    state.waiting_total = sum(state.waiting.values())
    snapshot = {
        "tick": state.tick,
        "occupied_total": sum(state.occupied.values()),
        "waiting_total": state.waiting_total,
        "nurses_used": total_nurses_used,
        "bed_utilization": (
            sum(state.occupied.values()) / total_beds if total_beds else 0.0
        ),
        "score": state.score,
    }
    state.history.append(snapshot)
    return state


def poisson_sample(rng: random.Random, mean: float) -> int:
    """Return a small Poisson sample using Knuth's algorithm."""
    import math

    limit = math.exp(-mean)
    count = 0
    product = 1.0
    while product > limit:
        count += 1
        product *= rng.random()
    return count - 1


def simulation_snapshot(
    state: SimulationState,
) -> list[dict[str, object]]:
    """Convert state into a dashboard-friendly department table."""
    rows = []
    for department in DEPARTMENTS:
        occupied = state.occupied[department]
        rows.append(
            {
                "department": department,
                "occupied_beds": occupied,
                "waiting": state.waiting[department],
                "nurses_used": nurse_requirement(occupied, department),
                "urgency_weight": URGENCY_WEIGHTS[department],
                "data_type": "synthetic",
            }
        )
    return rows

