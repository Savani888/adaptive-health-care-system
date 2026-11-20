"""Shared assumptions for the synthetic healthcare operations project."""

from pathlib import Path

# Use a fixed seed so the synthetic data and every result can be reproduced.
RANDOM_SEED = 42

# Generate three complete calendar years of daily observations.
DATA_START = "2023-01-01"
DATA_END = "2025-12-31"

# These are average daily admissions before calendar and outbreak effects.
DEPARTMENTS = (
    "Emergency",
    "Cardiology",
    "General Medicine",
    "Orthopaedics",
)
BASE_DAILY_ADMISSIONS = {
    "Emergency": 34,
    "Cardiology": 18,
    "General Medicine": 30,
    "Orthopaedics": 16,
}

# Admissions vary by day of week to make the data more realistic.
WEEKDAY_ADMISSION_MULTIPLIERS = {
    "Emergency": (1.04, 1.00, 0.99, 0.99, 1.00, 0.98, 1.02),
    "Cardiology": (0.94, 1.00, 1.04, 1.05, 1.03, 0.98, 0.96),
    "General Medicine": (0.94, 1.02, 1.04, 1.03, 1.02, 0.98, 0.97),
    "Orthopaedics": (0.90, 1.03, 1.07, 1.06, 1.04, 0.96, 0.94),
}

# Winter covers December through February in this synthetic example.
WINTER_MONTHS = (12, 1, 2)
WINTER_ADMISSION_MULTIPLIER = {
    "Emergency": 1.12,
    "Cardiology": 1.08,
    "General Medicine": 1.18,
    "Orthopaedics": 1.05,
}

# Admissions grow by 5% for each elapsed year.
YEARLY_ADMISSION_GROWTH = 0.05

# Add two synthetic outbreak waves. The last wave is deliberately within the
# last eight weeks of the generated series so backtests contain outbreak days.
OUTBREAK_WAVES = (
    {"start": "2024-01-10", "end": "2024-01-24", "multiplier": 1.30},
    {"start": "2025-11-25", "end": "2025-12-08", "multiplier": 1.45},
)
OUTBREAK_DEPARTMENT_MULTIPLIERS = {
    "Emergency": 1.25,
    "Cardiology": 1.05,
    "General Medicine": 1.20,
    "Orthopaedics": 0.95,
}

# Length of stay is sampled per admitted patient from a Poisson distribution,
# with a minimum of one day. This intentionally simple assumption is not
# intended to represent real hospital clinical pathways.
MEAN_LENGTH_OF_STAY_DAYS = {
    "Emergency": 2.0,
    "Cardiology": 5.0,
    "General Medicine": 4.0,
    "Orthopaedics": 4.5,
}

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
DEFAULT_DATA_PATH = DATA_DIR / "synthetic_daily_data.csv"

# Forecast models are trained separately for each department and lead time.
FORECAST_HORIZONS_DAYS = tuple(range(1, 8))
FORECAST_LAGS_DAYS = (0, 1, 2, 6, 13, 27)
FORECAST_ROLLING_WINDOWS_DAYS = (7, 28)
FORECAST_MIN_TRAINING_ROWS = 60
# A moderate ensemble keeps the daily refresh practical on a student laptop.
FORECAST_N_ESTIMATORS = 50
FORECAST_MAX_DEPTH = 2
FORECAST_MIN_SAMPLES_LEAF = 5
FORECAST_LEARNING_RATE = 0.05
FORECAST_LOSS = "huber"
FORECAST_TEST_DAYS = 56

# Allocation assumptions (all are synthetic planning parameters).
# A nurse ratio is the number of patients that one nurse may serve in a department.
NURSE_PATIENT_RATIOS = {
    "Emergency": 4,
    "Cardiology": 5,
    "General Medicine": 6,
    "Orthopaedics": 5,
}

# Higher weights express greater urgency. Emergency is highest by construction.
URGENCY_WEIGHTS = {
    "Emergency": 5,
    "Cardiology": 4,
    "General Medicine": 3,
    "Orthopaedics": 3,
}

# Default one-day planning capacities. Also used by the Streamlit allocation tab.
DEFAULT_TOTAL_BEDS = 240
DEFAULT_TOTAL_NURSES = 55

# Block size used later for the quantum QUBO approximation.
QUBO_BED_BLOCK_SIZE = 10

# Quantum-simulator limits. These keep the exact state vector at 18 qubits
# or fewer and are intentionally small demonstration instances.
QUANTUM_MAX_QUBITS = 18
QUANTUM_QAOA_SHOTS = 4_000
QUANTUM_RANDOM_SHOTS = 4_000
QUANTUM_ANNEALING_STEPS = 20_000

# Rolling adaptive-backtest assumptions.
ADAPTIVE_BACKTEST_DAYS = 56
FORECAST_SAFETY_MARGIN = 0.10
ADAPTIVE_RETRAIN_INTERVAL_DAYS = 7

# Synthetic real-time game simulation assumptions.
SIMULATION_SEED = 2026
SIMULATION_INITIAL_OCCUPANCY = {
    "Emergency": 55,
    "Cardiology": 42,
    "General Medicine": 68,
    "Orthopaedics": 38,
}
SIMULATION_ARRIVAL_PER_TICK = {
    "Emergency": 7,
    "Cardiology": 4,
    "General Medicine": 6,
    "Orthopaedics": 3,
}
SIMULATION_DISCHARGE_PROBABILITY = {
    "Emergency": 0.18,
    "Cardiology": 0.12,
    "General Medicine": 0.14,
    "Orthopaedics": 0.16,
}
