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
