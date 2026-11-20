"""Block-level QUBO, simulated annealing, and NumPy statevector QAOA."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from itertools import product
from math import ceil, log2

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from config import (
    DEPARTMENTS,
    QUBO_BED_BLOCK_SIZE,
    QUANTUM_ANNEALING_STEPS,
    QUANTUM_MAX_QUBITS,
    QUANTUM_QAOA_SHOTS,
    QUANTUM_RANDOM_SHOTS,
    RESULTS_DIR,
    URGENCY_WEIGHTS,
    NURSE_PATIENT_RATIOS,
)


@dataclass
class QuboInstance:
    """A hand-built quadratic binary objective."""

    linear: dict[int, float]
    quadratic: dict[tuple[int, int], float]
    constant: float
    labels: list[str]
    bed_items: dict[int, str]
    item_bed_weights: dict[int, int]
    item_nurse_weights: dict[int, int]
    bed_slack_start: int
    nurse_slack_start: int
    bed_slack_weights: list[int]
    nurse_slack_weights: list[int]
    feasible_beds: int
    feasible_nurses: int

    def evaluate(self, bits: np.ndarray) -> float:
        value = self.constant
        for i, coefficient in self.linear.items():
            value += coefficient * bits[i]
        for (i, j), coefficient in self.quadratic.items():
            value += coefficient * bits[i] * bits[j]
        return float(value)


def _binary_slack_bits(capacity: int) -> list[int]:
    """Create exact-range binary weights for a capacity."""
    if capacity < 0:
        raise ValueError("Capacity must be non-negative.")
    weights: list[int] = []
    remaining = capacity
    while remaining > 0:
        power = 1 << (remaining.bit_length() - 1)
        weights.append(power)
        remaining -= power
    return weights


def build_qubo(
    demand: dict[str, float | int],
    total_beds: int,
    total_nurses: int,
    penalty: float | None = None,
    block_size: int = QUBO_BED_BLOCK_SIZE,
) -> tuple[QuboInstance, dict[str, object]]:
    """Convert one block allocation into an exact-range QUBO."""
    if block_size <= 0:
        raise ValueError("Block size must be positive.")

    bed_blocks = max(0, total_beds // block_size)
    nurse_blocks = max(0, total_nurses)

    labels = []
    item_coefficients = {}
    item_bed_weights = {}
    item_nurse_weights = {}
    next_index = 0
    previous_item: dict[str, int] = {}
    for department in DEPARTMENTS:
        demand_blocks = int(demand[department]) // block_size
        for block in range(1, demand_blocks + 1):
            i = next_index
            next_index += 1
            labels.append(f"{department}_block_{block}")
            item_coefficients[i] = department
            item_bed_weights[i] = block_size
            item_nurse_weights[i] = (
                ceil(block * block_size / NURSE_PATIENT_RATIOS[department])
                - ceil((block - 1) * block_size / NURSE_PATIENT_RATIOS[department])
            )
            previous_item[department] = i

    bed_weights = _binary_slack_bits(total_beds)
    for offset, weight in enumerate(bed_weights):
        i = next_index
        next_index += 1
        labels.append(f"bed_slack_{offset}_{weight}")

    nurse_weights = _binary_slack_bits(total_nurses)
    for offset, weight in enumerate(nurse_weights):
        i = next_index
        next_index += 1
        labels.append(f"nurse_slack_{offset}_{weight}")

    n_qubits = len(labels)
    if n_qubits > QUANTUM_MAX_QUBITS:
        raise ValueError(
            f"QUBO needs {n_qubits} qubits; limit is {QUANTUM_MAX_QUBITS}."
        )

    if penalty is None:
        penalty = max(URGENCY_WEIGHTS.values()) * 20.0

    linear: dict[int, float] = {}
    quadratic: dict[tuple[int, int], float] = {}

    # The block objective rewards each separate block with urgency weight.
    for i, department in enumerate(item_coefficients.values()):
        service = block_size * URGENCY_WEIGHTS[department]
        linear[i] = linear.get(i, 0.0) - service

    # Selecting block k requires every earlier block k-1, k-2, ...:
    # P*(x_k - x_k*x_previous). This keeps nurse costs and allocated beds exact.
    for department in DEPARTMENTS:
        department_indices = [
            i for i, dep in item_coefficients.items() if dep == department
        ]
        for index_position, i in enumerate(department_indices):
            if index_position > 0:
                previous = department_indices[index_position - 1]
                linear[i] = linear.get(i, 0.0) + penalty
                pair = (previous, i)
                quadratic[pair] = quadratic.get(pair, 0.0) - penalty

    # Add bed and nurse capacity penalties in expanded quadratic form.

    item_indices = list(item_coefficients)
    bed_slack_start = len(item_coefficients)
    nurse_slack_start = bed_slack_start + len(bed_weights)

    def add_squared_capacity(
        item_weights: dict[int, int],
        slack_weights: list[int],
        slack_start: int,
        capacity: int,
    ) -> None:
        terms = [(i, w) for i, w in item_weights.items()]
        terms += [(slack_start + j, w) for j, w in enumerate(slack_weights)]
        linear_capacity = {i: w for i, w in terms}
        constant = capacity * capacity
        # Expand (sum(w_i x_i) - capacity)^2.
        for i, w in linear_capacity.items():
            linear[i] = linear.get(i, 0.0) + penalty * (w * w - 2.0 * w * capacity)
        for pos, (i, wi) in enumerate(terms):
            for j, wj in terms[pos + 1 :]:
                quadratic_key = (min(i, j), max(i, j))
                quadratic[quadratic_key] = (
                    quadratic.get(quadratic_key, 0.0) + 2.0 * penalty * wi * wj
                )
        _constant_holder.append(penalty * constant)

    _constant_holder: list[float] = []
    add_squared_capacity(item_bed_weights, bed_weights, bed_slack_start, total_beds)
    add_squared_capacity(item_nurse_weights, nurse_weights, nurse_slack_start, total_nurses)
    constant = sum(_constant_holder)

    qubo = QuboInstance(
        linear=linear,
        quadratic=quadratic,
        constant=constant,
        labels=labels,
        bed_items=dict(item_coefficients),
        item_bed_weights=item_bed_weights,
        item_nurse_weights=item_nurse_weights,
        bed_slack_start=bed_slack_start,
        nurse_slack_start=nurse_slack_start,
        bed_slack_weights=bed_weights,
        nurse_slack_weights=nurse_weights,
        feasible_beds=bed_blocks,
        feasible_nurses=nurse_blocks,
    )
    metadata = {
        "num_qubits": n_qubits,
        "block_size": block_size,
        "bed_block_capacity": bed_blocks,
        "nurse_capacity": total_nurses,
        "penalty": penalty,
        "item_variables": len(item_coefficients),
        "bed_slack_variables": len(bed_weights),
        "nurse_slack_variables": len(nurse_weights),
    }
    return qubo, metadata


def decode_solution(
    qubo: QuboInstance,
    bits: np.ndarray,
    block_size: int = QUBO_BED_BLOCK_SIZE,
) -> dict[str, object]:
    """Decode a basis state into block allocation and feasibility."""
    allocated: dict[str, int] = {department: 0 for department in DEPARTMENTS}
    bed_use = 0
    nurse_use = 0
    for department in DEPARTMENTS:
        indices = [
            i for i, department_name in qubo.bed_items.items()
            if department_name == department
        ]
        allocated_blocks = 0
        for i in indices:
            if i < len(bits) and bits[i] == 1:
                allocated_blocks += 1
            else:
                break
        allocated[department] = allocated_blocks * block_size
        bed_use += allocated_blocks * block_size
        nurse_use += ceil(
            allocated_blocks * block_size / NURSE_PATIENT_RATIOS[department]
        )
    return {
        "allocated": allocated,
        "bed_use": bed_use,
        "nurse_use": nurse_use,
        "feasible": bed_use <= qubo.feasible_beds * block_size
        and nurse_use <= qubo.feasible_nurses,
    }


def brute_force(qubo: QuboInstance) -> tuple[float, np.ndarray, int, float]:
    """Enumerate all states and return optimum plus exact feasible optimum."""
    best_energy = np.inf
    best_bits = np.zeros(len(qubo.labels), dtype=np.int8)
    feasible_count = 0
    feasible_optimum = np.inf
    for state in product((0, 1), repeat=len(qubo.labels)):
        bits = np.asarray(state, dtype=np.int8)
        energy = qubo.evaluate(bits)
        if energy < best_energy:
            best_energy = energy
            best_bits = bits.copy()
        decoded = decode_solution(
            qubo,
            bits,
            block_size=QUBO_BED_BLOCK_SIZE,
        )
        if decoded["feasible"]:
            feasible_count += 1
            if energy < feasible_optimum:
                feasible_optimum = energy
    return float(best_energy), best_bits, feasible_count, float(feasible_optimum)


def simulated_annealing(
    qubo: QuboInstance,
    seed: int = 42,
    steps: int = QUANTUM_ANNEALING_STEPS,
) -> tuple[float, np.ndarray, float]:
    """Run a classical Metropolis heuristic for the QUBO."""
    rng = np.random.default_rng(seed)
    n = len(qubo.labels)
    bits = rng.integers(0, 2, n, dtype=np.int8)
    current = qubo.evaluate(bits)
    best = current
    best_bits = bits.copy()
    start = time.perf_counter()
    for step in range(steps):
        temperature = max(0.01, 1.0 * (1.0 - step / steps))
        i = int(rng.integers(n))
        candidate = bits.copy()
        candidate[i] ^= 1
        candidate_energy = qubo.evaluate(candidate)
        difference = candidate_energy - current
        if difference <= 0 or rng.random() < np.exp(-difference / temperature):
            bits = candidate
            current = candidate_energy
            if current < best:
                best = current
                best_bits = bits.copy()
    return float(best), best_bits, time.perf_counter() - start


def random_sampling(qubo: QuboInstance, seed: int = 42, shots: int = QUANTUM_RANDOM_SHOTS) -> dict[str, object]:
    """Sample uniformly for a classical random-baseline comparison."""
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    bits = rng.integers(0, 2, size=(shots, len(qubo.labels)), dtype=np.int8)
    energies = np.array([qubo.evaluate(row) for row in bits])
    index = int(np.argmin(energies))
    return {
        "best_energy": float(energies[index]),
        "best_bits": bits[index],
        "runtime": time.perf_counter() - start,
    }


def _statevector_probabilities(
    qubo: QuboInstance,
    probabilities_or_angles: np.ndarray,
    precomputed_energies: np.ndarray | None = None,
) -> np.ndarray:
    """Apply p=1-3 QAOA layers to |+> and return bitstring probabilities."""
    n = len(qubo.labels)
    state = np.ones(2**n, dtype=np.complex128) / np.sqrt(2**n)
    layers = probabilities_or_angles.size // 2
    gammas = probabilities_or_angles[:layers]
    betas = probabilities_or_angles[layers:]

    diagonal = np.array(
        [qubo.evaluate(np.asarray(list(np.binary_repr(i, width=n)), dtype=np.int8))]
        for i in ()
    ) if False else None
    if precomputed_energies is None:
        energies = np.empty(2**n)
        for i in range(2**n):
            bits = np.array(
                [(i >> (n - 1 - j)) & 1 for j in range(n)],
                dtype=np.int8,
            )
            energies[i] = qubo.evaluate(bits)
    else:
        energies = precomputed_energies

    for gamma, beta in zip(gammas, betas):
        state *= np.exp(-1j * gamma * energies)
        state = _mix_x(state, beta)
    probabilities = np.abs(state) ** 2
    return probabilities / probabilities.sum()


def _mix_x(state: np.ndarray, beta: float) -> np.ndarray:
    """Apply exp(-i beta sum X) to a statevector."""
    n = int(np.log2(len(state)))
    for qubit in range(n):
        mixed = np.empty_like(state)
        for i in range(len(state)):
            partner = i ^ (1 << (n - 1 - qubit))
            mixed[i] = np.cos(beta) * state[i] - 1j * np.sin(beta) * state[partner]
        state = mixed
    return state


def run_qaoa(
    qubo: QuboInstance,
    p: int,
    shots: int = QUANTUM_QAOA_SHOTS,
    seed: int = 42,
) -> dict[str, object]:
    """Optimize QAOA angles and sample basis states."""
    if p not in (1, 2, 3):
        raise ValueError("p must be 1, 2, or 3.")
    n = len(qubo.labels)
    if n > 18:
        raise ValueError("QAOA statevector limit exceeded.")
    state_count = 2**n
    state_energies = np.empty(state_count)
    for i in range(state_count):
        bits = np.array(
            [(i >> (n - 1 - j)) & 1 for j in range(n)],
            dtype=np.int8,
        )
        state_energies[i] = qubo.evaluate(bits)

    # Multiple starts keep the small-angle search more reliable; COBYLA is
    # fast enough for the 1-to-3-layer statevector demonstrations.
    def objective(angles: np.ndarray) -> float:
        probabilities = _statevector_probabilities(
            qubo,
            angles,
            precomputed_energies=state_energies,
        )
        return float(np.sum(probabilities * state_energies))

    bounds = [(0.0, 2 * np.pi)] * p + [(0.0, np.pi)] * p
    start = time.perf_counter()
    optimized = minimize(
        objective,
        np.array([np.pi / 4] * p + [np.pi / 8] * p),
        method="Nelder-Mead",
        options={"maxiter": 30, "xatol": 1e-3, "fatol": 1e-3},
    )
    runtime = time.perf_counter() - start
    probabilities = _statevector_probabilities(
        qubo,
        optimized.x,
        precomputed_energies=state_energies,
    )
    rng = np.random.default_rng(seed)
    samples = rng.choice(2**n, size=shots, p=probabilities)
    energies = []
    optima = 0
    optimum_energy = brute_force(qubo)[0]
    feasible_shots = 0
    sampled_best_objective = -np.inf
    true_optimum_objective = 0.0
    for i in range(2**n):
        decoded = decode_solution(
            qubo,
            np.array([(i >> (n - 1 - j)) & 1 for j in range(n)], dtype=np.int8),
        )
        if decoded["feasible"]:
            objective_value = sum(
                QUBO_BED_BLOCK_SIZE * URGENCY_WEIGHTS[department]
                * (decoded["allocated"][department] // QUBO_BED_BLOCK_SIZE)
                for department in DEPARTMENTS
            )
            true_optimum_objective = max(true_optimum_objective, objective_value)
    for index in samples:
        bits = np.array(
            [(int(index) >> (n - 1 - j)) & 1 for j in range(n)],
            dtype=np.int8,
        )
        energy = qubo.evaluate(bits)
        energies.append(energy)
        decoded = decode_solution(qubo, bits)
        if decoded["feasible"]:
            feasible_shots += 1
            objective_value = sum(
                QUBO_BED_BLOCK_SIZE * URGENCY_WEIGHTS[department]
                * (decoded["allocated"][department] // QUBO_BED_BLOCK_SIZE)
                for department in DEPARTMENTS
            )
            sampled_best_objective = max(sampled_best_objective, objective_value)
        if np.isclose(energy, optimum_energy):
            optima += 1
    approximation_ratio = (
        sampled_best_objective / true_optimum_objective
        if true_optimum_objective > 0 and np.isfinite(sampled_best_objective)
        else 0.0
    )
    return {
        "p": p,
        "num_qubits": n,
        "runtime": runtime,
        "probabilities": probabilities,
        "best_sample_energy": float(min(energies)),
        "optimum_probability": float(probabilities[
            np.argmin(
                [
                    qubo.evaluate(
                        np.array(
                            [(i >> (n - 1 - j)) & 1 for j in range(n)],
                            dtype=np.int8,
                        )
                    )
                    for i in range(2**n)
                ]
            )
        ]),
        "optimum_sample_probability": optima / shots,
        "feasible_shot_rate": feasible_shots / shots,
        "approximation_ratio": approximation_ratio,
        "angles": optimized.x,
    }


def validate_matrix_exponential_tiny() -> bool:
    """Compare NumPy statevector QAOA against scipy.linalg.expm for one qubit."""
    qubo = QuboInstance(
        linear={0: -1.0},
        quadratic={},
        constant=0.0,
        labels=["x"],
        bed_items={0: "Emergency"},
        item_bed_weights={0: 10},
        item_nurse_weights={0: 3},
        bed_slack_start=1,
        nurse_slack_start=1,
        bed_slack_weights=[],
        nurse_slack_weights=[],
        feasible_beds=1,
        feasible_nurses=10,
    )
    gamma = 0.23
    beta = 0.41
    state = np.ones(2, dtype=np.complex128) / np.sqrt(2)
    state *= np.exp(-1j * gamma * np.array([qubo.evaluate(np.array([0])), qubo.evaluate(np.array([1]))]))
    state = np.array(
        [[np.cos(beta), -1j * np.sin(beta)],
         [-1j * np.sin(beta), np.cos(beta)]]
    ) @ state
    mixer = np.array([[0, 1], [1, 0]], dtype=float)
    expected = expm(-1j * beta * mixer) @ (
        expm(-1j * gamma * np.diag([qubo.evaluate(np.array([0])), qubo.evaluate(np.array([1]))]))
        @ (np.ones(2) / np.sqrt(2))
    )
    return bool(np.allclose(state, expected, atol=1e-10))


def main() -> None:
    """Run a one-day block allocation comparison and save real synthetic results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beds", type=int, default=20)
    parser.add_argument("--nurses", type=int, default=10)
    args = parser.parse_args()
    demand = {
        "Emergency": 10,
        "Cardiology": 10,
        "General Medicine": 10,
        "Orthopaedics": 10,
    }
    qubo, metadata = build_qubo(demand, args.beds, args.nurses)
    optimum, optimum_bits, _, _ = brute_force(qubo)
    annealed, annealed_bits, annealed_runtime = simulated_annealing(qubo)
    random = random_sampling(qubo)
    matrix_valid = validate_matrix_exponential_tiny()

    records = [{
        "solver": "Brute force",
        "energy": optimum,
        "runtime_seconds": 0.0,
        "optimum_probability": 1.0,
        "data_type": "synthetic",
    }, {
        "solver": "Simulated annealing",
        "energy": annealed,
        "runtime_seconds": annealed_runtime,
        "optimum_probability": 0.0,
        "data_type": "synthetic",
    }, {
        "solver": "Random sampling",
        "energy": random["best_energy"],
        "runtime_seconds": random["runtime"],
        "optimum_probability": 0.0,
        "data_type": "synthetic",
    }]
    for p in (1, 2, 3):
        qaoa = run_qaoa(qubo, p)
        records.append({
            "solver": f"QAOA p={p}",
            "energy": qaoa["best_sample_energy"],
            "runtime_seconds": qaoa["runtime"],
            "optimum_probability": qaoa["optimum_sample_probability"],
            "feasible_shot_rate": qaoa["feasible_shot_rate"],
            "approximation_ratio": qaoa["approximation_ratio"],
            "data_type": "synthetic",
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "quantum_comparison.csv"
    pd = __import__("pandas")
    pd.DataFrame(records).to_csv(path, index=False)
    print(pd.DataFrame(records).to_string(index=False))
    print("Metadata:", metadata)
    print("Matrix-exponential tiny validation:", matrix_valid)
    print("Saved:", path)


if __name__ == "__main__":
    main()
