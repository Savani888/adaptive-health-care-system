"""Tests for the real-time synthetic hospital game."""

from simulation import (
    initialize_simulation,
    nurse_requirement,
    simulation_snapshot,
    simulate_tick,
)


def test_simulation_tick_enforces_beds_and_nurses() -> None:
    state = initialize_simulation(seed=42)
    before_occupied = sum(state.occupied.values())
    simulate_tick(state, total_beds=100, total_nurses=25, surge_percent=50)

    assert state.tick == 1
    assert sum(state.occupied.values()) <= 100
    nurses_used = sum(
        nurse_requirement(state.occupied[department], department)
        for department in state.occupied
    )
    assert nurses_used <= 25
    assert state.waiting_total >= 0
    assert len(state.history) == 1
    assert before_occupied >= 0


def test_snapshot_is_synthetic_and_covers_four_departments() -> None:
    state = initialize_simulation(seed=7)
    snapshot = simulation_snapshot(state)
    assert len(snapshot) == 4
    assert {row["department"] for row in snapshot} == set(state.occupied)
    assert all(row["data_type"] == "synthetic" for row in snapshot)
