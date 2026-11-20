"""Tests for the block QUBO and NumPy statevector QAOA."""

import numpy as np

from quantum import (
    brute_force,
    build_qubo,
    decode_solution,
    run_qaoa,
    validate_matrix_exponential_tiny,
)


def test_tiny_qubo_matches_brute_force_and_matrix_exponential() -> None:
    qubo, metadata = build_qubo(
        {
            "Emergency": 20,
            "Cardiology": 20,
            "General Medicine": 10,
            "Orthopaedics": 10,
        },
        total_beds=20,
        total_nurses=10,
    )
    optimum_energy, _, feasible_count, feasible_optimum = brute_force(qubo)
    assert metadata["num_qubits"] <= 18
    assert feasible_count > 0
    assert np.isfinite(feasible_optimum)
    assert optimum_energy <= feasible_optimum
    assert validate_matrix_exponential_tiny()


def test_qaoa_probabilities_sum_to_one() -> None:
    qubo, _ = build_qubo(
        {
            "Emergency": 10,
            "Cardiology": 10,
            "General Medicine": 10,
            "Orthopaedics": 10,
        },
        total_beds=20,
        total_nurses=10,
    )
    for p in (1, 2, 3):
        result = run_qaoa(qubo, p=p, shots=500, seed=42)
        probabilities = result["probabilities"]
        assert np.isclose(probabilities.sum(), 1.0)
        assert (probabilities >= 0).all()
        assert 0 <= result["optimum_probability"] <= 1


def test_decoded_solution_obeys_bed_and_nurse_capacity() -> None:
    qubo, _ = build_qubo(
        {
            "Emergency": 20,
            "Cardiology": 10,
            "General Medicine": 10,
            "Orthopaedics": 10,
        },
        total_beds=20,
        total_nurses=8,
    )
    _, bits, _, _ = brute_force(qubo)
    decoded = decode_solution(qubo, bits)
    assert decoded["bed_use"] >= 0
    assert decoded["nurse_use"] >= 0
