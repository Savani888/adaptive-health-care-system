"""Streamlit dashboard for the adaptive healthcare operations project."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from allocation import compare_allocations
from config import (
    DEPARTMENTS,
    DEFAULT_TOTAL_BEDS,
    DEFAULT_TOTAL_NURSES,
    NURSE_PATIENT_RATIOS,
    PROJECT_DIR,
    URGENCY_WEIGHTS,
)

st.set_page_config(
    page_title="Adaptive Healthcare Operations",
    page_icon="H",
    layout="wide",
)

DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    """Load a saved CSV once while Streamlit rerenders widgets."""
    return pd.read_csv(path)


def forecast_tab() -> None:
    st.subheader("Seven-day synthetic census outlook")
    forecast_path = RESULTS_DIR / "seven_day_forecast.csv"
    if not forecast_path.exists():
        st.warning("Run forecasting.py to create seven_day_forecast.csv.")
        return
    forecast = load_csv(str(forecast_path))
    department = st.selectbox("Department", DEPARTMENTS)
    department_forecast = forecast.loc[
        forecast["department"] == department
    ].copy()
    fig = px.line(
        department_forecast,
        x="date",
        y=["forecast_census", "baseline_last_week", "baseline_28day_mean"],
        title=f"Synthetic forecast: {department}",
        labels={"value": "Patients", "date": "Date", "variable": "Series"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        forecast.loc[
            forecast["department"] == department
        ].drop(columns=["data_type"]),
        use_container_width=True,
    )

    metrics_path = RESULTS_DIR / "forecast_metrics.csv"
    if metrics_path.exists():
        metrics = load_csv(str(metrics_path))
        st.caption("Holdout error by horizon")
        st.dataframe(
            metrics.loc[
                metrics["department"] == department,
                [
                    "horizon",
                    "model_mae",
                    "model_rmse",
                    "model_mape_pct",
                    "last_week_mae",
                    "mean_28day_mae",
                ],
            ],
            use_container_width=True,
        )


def allocation_tab() -> None:
    st.subheader("Bed and nurse allocation")
    forecast_path = RESULTS_DIR / "seven_day_forecast.csv"
    if forecast_path.exists():
        forecast = load_csv(str(forecast_path))
        horizon_1 = forecast.loc[forecast["horizon"] == 1].copy()
        default_demand = dict(
            zip(horizon_1["department"], horizon_1["forecast_census"])
        )
    else:
        default_demand = {department: 50.0 for department in DEPARTMENTS}

    beds = st.slider("Available beds", 0, 500, DEFAULT_TOTAL_BEDS)
    nurses = st.slider("Available nurses", 0, 120, DEFAULT_TOTAL_NURSES)
    surge = st.slider("Demand surge", 0, 50, 0, format="%d%%")

    demand = {
        department: float(
            default_demand.get(department, 50.0) * (1.0 + surge / 100.0)
        )
        for department in DEPARTMENTS
    }
    try:
        ilp, dp = compare_allocations(demand, beds, nurses)
        table = []
        for result in (ilp, dp):
            for department in DEPARTMENTS:
                table.append(
                    {
                        "method": result.method,
                        "department": department,
                        "patients_served": result.patients_served[department],
                        "nurses": result.nurse_allocations[department],
                        "forecast_demand": demand[department],
                        "data_type": "synthetic",
                    }
                )
        allocation = pd.DataFrame(table)
        st.dataframe(
            allocation.drop(columns=["data_type"]),
            use_container_width=True,
        )
        fig = px.bar(
            allocation,
            x="department",
            y="patients_served",
            color="method",
            barmode="group",
            title="Synthetic patients served",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Matched weighted objective", str(int(ilp.objective)))
    except Exception as error:
        st.error(f"Allocation could not be solved: {error}")


def quantum_tab() -> None:
    st.subheader("Quantum vs classical comparison")
    if st.button("Run synthetic 8-qubit comparison"):
        from quantum import (
            brute_force,
            build_qubo,
            random_sampling,
            run_qaoa,
            simulated_annealing,
        )

        with st.spinner("Running NumPy statevector and classical solvers..."):
            demand = {
                "Emergency": 10,
                "Cardiology": 10,
                "General Medicine": 10,
                "Orthopaedics": 10,
            }
            qubo, metadata = build_qubo(demand, 20, 10)
            optimum, _, _, _ = brute_force(qubo)
            annealed, _, annealed_time = simulated_annealing(qubo)
            random = random_sampling(qubo)
            records = [
                {
                    "solver": "Brute force",
                    "energy": optimum,
                    "runtime_seconds": 0.0,
                    "data_type": "synthetic",
                },
                {
                    "solver": "Simulated annealing",
                    "energy": annealed,
                    "runtime_seconds": annealed_time,
                    "data_type": "synthetic",
                },
                {
                    "solver": "Random sampling",
                    "energy": random["best_energy"],
                    "runtime_seconds": random["runtime"],
                    "data_type": "synthetic",
                },
            ]
            for p in (1, 2, 3):
                result = run_qaoa(qubo, p=p)
                records.append(
                    {
                        "solver": f"QAOA p={p}",
                        "energy": result["best_sample_energy"],
                        "runtime_seconds": result["runtime"],
                        "optimum_sample_probability": result[
                            "optimum_sample_probability"
                        ],
                        "feasible_shot_rate": result["feasible_shot_rate"],
                        "approximation_ratio": result["approximation_ratio"],
                        "data_type": "synthetic",
                    }
                )
            st.dataframe(pd.DataFrame(records), use_container_width=True)
            st.caption(
                f"Synthetic NumPy simulator: {metadata['num_qubits']} qubits. "
                "No quantum-advantage claim."
            )

    saved_path = RESULTS_DIR / "quantum_comparison.csv"
    if saved_path.exists():
        st.caption("Previously saved quantum comparison")
        st.dataframe(load_csv(str(saved_path)), use_container_width=True)


def backtest_tab() -> None:
    st.subheader("Rolling backtest")
    summary_path = RESULTS_DIR / "adaptive_backtest_summary.csv"
    if not summary_path.exists():
        st.warning("Run adaptive.py to create adaptive_backtest_summary.csv.")
        return
    summary = load_csv(str(summary_path))
    fig = px.bar(
        summary,
        x="strategy",
        y="weighted_service_level",
        color="period",
        barmode="group",
        title="Synthetic weighted service level",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(summary, use_container_width=True)


def about_tab() -> None:
    st.subheader("About this synthetic project")
    st.markdown(
        """
        **Data:** Three years of synthetic daily admissions and census. No real patient
        records or database are used.

        **Forecast:** GradientBoostingRegressor, one model per department and horizon.
        Training examples use only target dates already known.

        **Allocation:** Weighted integer optimization for beds and nurse capacity.
        Nurse ratios and urgency weights are assumptions, not clinical policy.

        **Quantum:** NumPy statevector QAOA for small block instances. Qiskit is optional;
        this implementation validates matrix-exponential behavior and does not claim
        quantum advantage.

        **Limitations:** Daily synthetic census is not a clinical model, nurse coverage
        is simplified to ratios, and the dashboard is a planning demonstration.
        """
    )
    st.dataframe(
        pd.DataFrame(
            {
                "department": DEPARTMENTS,
                "nurse_patients_per_nurse": [
                    NURSE_PATIENT_RATIOS[department]
                    for department in DEPARTMENTS
                ],
                "urgency_weight": [
                    URGENCY_WEIGHTS[department]
                    for department in DEPARTMENTS
                ],
            }
        ),
        use_container_width=True,
    )


def simulation_tab() -> None:
    """Render the synthetic live hospital game and controls."""
    from simulation import (
        initialize_simulation,
        simulation_snapshot,
        simulate_tick,
    )

    st.subheader("Real-time synthetic hospital operations game")
    st.caption(
        "Synthetic patients arrive and leave automatically. Adjust capacity and "
        "watch waiting queues, bed use, nurse use, and game score."
    )

    if "simulation_state" not in st.session_state:
        st.session_state["simulation_state"] = initialize_simulation()
    if "simulation_running" not in st.session_state:
        st.session_state["simulation_running"] = False

    left, right = st.columns([1, 3])
    with left:
        total_beds = st.slider(
            "Game beds",
            100,
            400,
            DEFAULT_TOTAL_BEDS,
            key="game_beds",
        )
        total_nurses = st.slider(
            "Game nurses",
            20,
            100,
            DEFAULT_TOTAL_NURSES,
            key="game_nurses",
        )
        surge = st.slider(
            "Arrival surge",
            0,
            100,
            0,
            format="%d%%",
            key="game_surge",
        )
        speed = st.select_slider(
            "Tick every",
            options=["1 second", "2 seconds", "5 seconds"],
            value="2 seconds",
            key="game_speed",
        )
        if st.button("Start live game"):
            st.session_state["simulation_running"] = True
        if st.button("Pause"):
            st.session_state["simulation_running"] = False
        if st.button("Reset synthetic hospital"):
            st.session_state["simulation_state"] = initialize_simulation()
            st.session_state["simulation_running"] = False
        if st.button("Advance one tick"):
            st.session_state["simulation_state"] = simulate_tick(
                st.session_state["simulation_state"],
                total_beds,
                total_nurses,
                surge,
            )

    @st.fragment(
        run_every={
            "1 second": 1,
            "2 seconds": 2,
            "5 seconds": 5,
        }[speed]
    )
    def live_fragment() -> None:
        if st.session_state["simulation_running"]:
            st.session_state["simulation_state"] = simulate_tick(
                st.session_state["simulation_state"],
                st.session_state["game_beds"],
                st.session_state["game_nurses"],
                st.session_state["game_surge"],
            )

        state = st.session_state["simulation_state"]
        snapshot = simulation_snapshot(state)
        table = pd.DataFrame(snapshot)
        occupied_total = sum(state.occupied.values())
        nurses_used = sum(row["nurses_used"] for row in snapshot)

        metric_left, metric_mid, metric_right, metric_score = st.columns(4)
        metric_left.metric("Synthetic tick", state.tick)
        metric_mid.metric("Occupied beds", f"{occupied_total}/{st.session_state['game_beds']}")
        metric_right.metric("Nurses used", f"{nurses_used}/{st.session_state['game_nurses']}")
        metric_score.metric("Game score", state.score)

        st.progress(
            min(1.0, occupied_total / max(1, st.session_state["game_beds"])),
            text="Bed utilization",
        )

        bars = px.bar(
            table,
            x="department",
            y=["occupied_beds", "waiting"],
            barmode="group",
            title="Occupied beds and waiting patients",
            labels={"value": "Patients", "variable": "Status"},
        )
        st.plotly_chart(bars, use_container_width=True)

        history = pd.DataFrame(state.history)
        if not history.empty:
            trend = px.line(
                history,
                x="tick",
                y=["occupied_total", "waiting_total", "nurses_used"],
                title="Live synthetic operations trend",
            )
            st.plotly_chart(trend, use_container_width=True)
        st.dataframe(
            table.drop(columns=["data_type"]),
            use_container_width=True,
        )
        if state.waiting_total:
            st.warning(
                f"{state.waiting_total} synthetic patients are currently waiting."
            )
        else:
            st.success("All synthetic arrivals were admitted.")

    with right:
        live_fragment()


def upload_tab() -> None:
    st.subheader("Optional synthetic CSV upload")
    uploaded = st.file_uploader(
        "Upload columns: date, department, admissions, census",
        type=["csv"],
    )
    if uploaded is not None:
        uploaded_data = pd.read_csv(uploaded)
        required = {"date", "department", "admissions", "census"}
        missing = required.difference(uploaded_data.columns)
        if missing:
            st.error(f"Missing columns: {sorted(missing)}")
        else:
            st.success("CSV structure accepted.")
            st.dataframe(uploaded_data.head(100), use_container_width=True)


st.title("Adaptive Healthcare Operations Intelligence")
st.caption("Synthetic data only - planning demonstration")

tabs = st.tabs(
    [
        "Forecast",
        "Allocation",
        "Quantum vs Classical",
        "Backtest",
        "Live Game",
        "About",
        "CSV Upload",
    ]
)

with tabs[0]:
    forecast_tab()
with tabs[1]:
    allocation_tab()
with tabs[2]:
    quantum_tab()
with tabs[3]:
    backtest_tab()
with tabs[4]:
    simulation_tab()
with tabs[5]:
    about_tab()
with tabs[6]:
    upload_tab()
