import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Page configuration
st.set_page_config(page_title="Sprint Time Predictor", page_icon="🏃‍♂️", layout="wide")

# --- INITIALIZE SESSION HISTORY MEMORY ---
if "saved_runs_100" not in st.session_state:
    st.session_state.saved_runs_100 = []
if "saved_runs_200" not in st.session_state:
    st.session_state.saved_runs_200 = []

# Load the trained models using Streamlit's caching
@st.cache_resource
def load_100m_model():
    try:
        return joblib.load("100m_model.pkl")
    except FileNotFoundError:
        return None

@st.cache_resource
def load_200m_model():
    try:
        return joblib.load("200m_model.pkl")
    except FileNotFoundError:
        return None

model_100m = load_100m_model()
model_200m = load_200m_model()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🏃‍♂️ Track Events")
event_choice = st.sidebar.radio("Select Event Predictor", ["100m Dash", "200m Dash"])

st.title(f"🏃‍♂️ {event_choice} Analytics Dashboard")
st.divider()

# Helper function to inject NaN if the user marked it as missing
def get_val(name, val, missing_list):
    return np.nan if name in missing_list else val

# =====================================================================
# 100M DASH EVENT LOGIC
# =====================================================================
if event_choice == "100m Dash":
    if model_100m is None:
        st.error("Model file '100m_model.pkl' not found. Please run train_and_save.py first.")
        st.stop()
        
    tab1, tab2 = st.tabs(["📊 Predictor & Analysis", "🎯 Goal Calculator"])

    with tab1:
        st.markdown("Adjust the sliders for the athlete's metrics to instantly predict their final 100m time and analyze their race phases.")
        
        st.subheader("Missing Data?")
        missing_metrics = st.multiselect(
            "Select any measurements you do NOT know. The AI will estimate the race without them:",
            [
                "Reaction Time", "Wind", 
                "Time 10m", "Time 20m", "Time 30m", "Time 40m", "Time 50m", "Time 60m", "Time 70m", "Time 80m", "Time 90m",
                "Velocity 10m", "Velocity 20m", "Velocity 30m", "Velocity 40m", "Velocity 50m", "Velocity 60m", "Velocity 70m", "Velocity 80m", "Velocity 90m"
            ],
            placeholder="e.g., Velocity 10m, Time 90m..."
        )
        
        with st.expander("ℹ️ How to Calculate Velocity Inputs"):
            st.markdown("""
            **Velocity = Distance ÷ Time**
            Because we are looking at 10-meter blocks, you need to find the *time spent in that specific 10m zone*. 
            **Example: Calculating Velocity at 20m**
            * **Time 10m:** 1.85 s
            * **Time 20m:** 2.90 s
            * **Time spent in this 10m zone:** 2.90 - 1.85 = **1.05 s**
            * **Calculation:** 10 meters ÷ 1.05 seconds = **9.52 m/s**
            """)
            
        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.header("Race Conditions")
            reaction_time = st.slider("Reaction Time (s)", min_value=0.100, max_value=0.400, value=0.150, step=0.001, format="%.3f", disabled=("Reaction Time" in missing_metrics))
            wind = st.slider("Wind (m/s)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1, format="%.1f", disabled=("Wind" in missing_metrics))

        with col2:
            st.header("Split Times (s)")
            t_10 = st.slider("Time 10m", min_value=1.00, max_value=4.00, value=1.85, step=0.01, format="%.2f", disabled=("Time 10m" in missing_metrics))
            t_20 = st.slider("Time 20m", min_value=2.00, max_value=6.00, value=2.90, step=0.01, format="%.2f", disabled=("Time 20m" in missing_metrics))
            t_30 = st.slider("Time 30m", min_value=3.00, max_value=8.00, value=3.80, step=0.01, format="%.2f", disabled=("Time 30m" in missing_metrics))
            t_40 = st.slider("Time 40m", min_value=4.00, max_value=10.00, value=4.65, step=0.01, format="%.2f", disabled=("Time 40m" in missing_metrics))
            t_50 = st.slider("Time 50m", min_value=4.50, max_value=12.00, value=5.50, step=0.01, format="%.2f", disabled=("Time 50m" in missing_metrics))
            t_60 = st.slider("Time 60m", min_value=5.50, max_value=14.00, value=6.35, step=0.01, format="%.2f", disabled=("Time 60m" in missing_metrics))
            t_70 = st.slider("Time 70m", min_value=6.00, max_value=16.00, value=7.20, step=0.01, format="%.2f", disabled=("Time 70m" in missing_metrics))
            t_80 = st.slider("Time 80m", min_value=7.00, max_value=18.00, value=8.10, step=0.01, format="%.2f", disabled=("Time 80m" in missing_metrics))
            t_90 = st.slider("Time 90m", min_value=8.00, max_value=20.00, value=9.00, step=0.01, format="%.2f", disabled=("Time 90m" in missing_metrics))

        with col3:
            st.header("Velocities (m/s)")
            v_10 = st.slider("Velocity 10m", min_value=0.0, max_value=15.0, value=5.40, step=0.01, format="%.2f", disabled=("Velocity 10m" in missing_metrics))
            v_20 = st.slider("Velocity 20m", min_value=0.0, max_value=15.0, value=8.60, step=0.01, format="%.2f", disabled=("Velocity 20m" in missing_metrics))
            v_30 = st.slider("Velocity 30m", min_value=0.0, max_value=15.0, value=10.20, step=0.01, format="%.2f", disabled=("Velocity 30m" in missing_metrics))
            v_40 = st.slider("Velocity 40m", min_value=0.0, max_value=15.0, value=11.10, step=0.01, format="%.2f", disabled=("Velocity 40m" in missing_metrics))
            v_50 = st.slider("Velocity 50m", min_value=0.0, max_value=15.0, value=11.50, step=0.01, format="%.2f", disabled=("Velocity 50m" in missing_metrics))
            v_60 = st.slider("Velocity 60m", min_value=0.0, max_value=15.0, value=11.75, step=0.01, format="%.2f", disabled=("Velocity 60m" in missing_metrics))
            v_70 = st.slider("Velocity 70m", min_value=0.0, max_value=15.0, value=11.60, step=0.01, format="%.2f", disabled=("Velocity 70m" in missing_metrics))
            v_80 = st.slider("Velocity 80m", min_value=0.0, max_value=15.0, value=11.30, step=0.01, format="%.2f", disabled=("Velocity 80m" in missing_metrics))
            v_90 = st.slider("Velocity 90m", min_value=0.0, max_value=15.0, value=11.10, step=0.01, format="%.2f", disabled=("Velocity 90m" in missing_metrics))

        st.divider()

        input_data = {
            "reaction time": [get_val("Reaction Time", reaction_time, missing_metrics)], 
            "wind": [get_val("Wind", wind, missing_metrics)],
            "time 10m": [get_val("Time 10m", t_10, missing_metrics)], "time 20m": [get_val("Time 20m", t_20, missing_metrics)], 
            "time 30m": [get_val("Time 30m", t_30, missing_metrics)], "time 40m": [get_val("Time 40m", t_40, missing_metrics)], 
            "time 50m": [get_val("Time 50m", t_50, missing_metrics)], "time 60m": [get_val("Time 60m", t_60, missing_metrics)], 
            "time 70m": [get_val("Time 70m", t_70, missing_metrics)], "time 80m": [get_val("Time 80m", t_80, missing_metrics)], 
            "time 90m": [get_val("Time 90m", t_90, missing_metrics)],
            "velocity 10m": [get_val("Velocity 10m", v_10, missing_metrics)], "velocity 20m": [get_val("Velocity 20m", v_20, missing_metrics)], 
            "velocity 30m": [get_val("Velocity 30m", v_30, missing_metrics)], "velocity 40m": [get_val("Velocity 40m", v_40, missing_metrics)], 
            "velocity 50m": [get_val("Velocity 50m", v_50, missing_metrics)], "velocity 60m": [get_val("Velocity 60m", v_60, missing_metrics)], 
            "velocity 70m": [get_val("Velocity 70m", v_70, missing_metrics)], "velocity 80m": [get_val("Velocity 80m", v_80, missing_metrics)], 
            "velocity 90m": [get_val("Velocity 90m", v_90, missing_metrics)]
        }

        input_df = pd.DataFrame(input_data)
        prediction = model_100m.predict(input_df)[0]

        if "Time 90m" in missing_metrics:
            v_100 = np.nan
        else:
            final_10m_time = prediction - t_90
            v_100 = 10 / final_10m_time if final_10m_time > 0 else v_90
            
        distances = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        times = [
            0.0, get_val("Time 10m", t_10, missing_metrics), get_val("Time 20m", t_20, missing_metrics), get_val("Time 30m", t_30, missing_metrics), 
            get_val("Time 40m", t_40, missing_metrics), get_val("Time 50m", t_50, missing_metrics), get_val("Time 60m", t_60, missing_metrics), 
            get_val("Time 70m", t_70, missing_metrics), get_val("Time 80m", t_80, missing_metrics), get_val("Time 90m", t_90, missing_metrics), prediction
        ]
        velocities = [
            0.0, get_val("Velocity 10m", v_10, missing_metrics), get_val("Velocity 20m", v_20, missing_metrics), get_val("Velocity 30m", v_30, missing_metrics), 
            get_val("Velocity 40m", v_40, missing_metrics), get_val("Velocity 50m", v_50, missing_metrics), get_val("Velocity 60m", v_60, missing_metrics), 
            get_val("Velocity 70m", v_70, missing_metrics), get_val("Velocity 80m", v_80, missing_metrics), get_val("Velocity 90m", v_90, missing_metrics), v_100
        ]
        
        accelerations = [0.0]
        for i in range(1, len(distances)):
            dt = times[i] - times[i-1]
            dv = velocities[i] - velocities[i-1]
            if pd.isna(dt) or pd.isna(dv) or dt <= 0:
                accelerations.append(np.nan)
            else:
                accelerations.append(dv / dt)

        # --- PREDICTION METRIC & SAVE RUN CONTROLS ---
        pred_col, save_col = st.columns([1, 1])
        
        with pred_col:
            st.metric(label="Predicted Finish Time", value=f"{prediction:.3f} s")
            
        with save_col:
            st.markdown("**Session Comparison**")
            run_label = st.text_input("Run Label / Athlete Name", placeholder="e.g., Heat 1 or Athlete A", label_visibility="collapsed")
            if st.button("💾 Save Run to 100m Session"):
                label_to_save = run_label.strip() if run_label.strip() else f"Run {len(st.session_state.saved_runs_100) + 1}"
                st.session_state.saved_runs_100.append({
                    "Run Label": label_to_save,
                    "Predicted Time (s)": round(prediction, 3),
                    "Reaction Time (s)": reaction_time if "Reaction Time" not in missing_metrics else "N/A",
                    "Wind (m/s)": wind if "Wind" not in missing_metrics else "N/A",
                    "Velocities": velocities,
                    "Accelerations": accelerations
                })
                st.success(f"Saved '{label_to_save}' to session history!")

        st.divider()

        # --- RACE PHASE BREAKDOWN ---
        st.header("⏱️ Race Phase Breakdown")
        if "Time 30m" in missing_metrics or "Time 60m" in missing_metrics:
            st.info("⚠️ Please uncheck 'Time 30m' and 'Time 60m' to view the Race Phase Breakdown.")
        else:
            drive_time = t_30
            max_v_time = t_60 - t_30
            endurance_time = prediction - t_60
            drive_vel = 30 / drive_time if drive_time > 0 else 0
            max_v_vel = 30 / max_v_time if max_v_time > 0 else 0
            endurance_vel = 40 / endurance_time if endurance_time > 0 else 0
            
            ph_col1, ph_col2, ph_col3 = st.columns(3)
            with ph_col1:
                st.metric("Drive Phase (0-30m)", f"{drive_time:.2f} s", f"{drive_vel:.2f} m/s avg", delta_color="off")
            with ph_col2:
                st.metric("Max Velocity (30-60m)", f"{max_v_time:.2f} s", f"{max_v_vel:.2f} m/s avg", delta_color="off")
            with ph_col3:
                st.metric("Speed Endurance (60-100m)", f"{endurance_time:.2f} s", f"{endurance_vel:.2f} m/s avg", delta_color="off")

        st.divider()

        # --- SESSION HISTORY TABLE ---
        if len(st.session_state.saved_runs_100) > 0:
            st.header("📋 100m Session History")
            history_df = pd.DataFrame([{ "Run Label": r["Run Label"], "Predicted 100m (s)": r["Predicted Time (s)"], "Reaction (s)": r["Reaction Time (s)"], "Wind (m/s)": r["Wind (m/s)"]} for r in st.session_state.saved_runs_100])
            tbl_col, btn_col = st.columns([3, 1])
            with tbl_col:
                st.dataframe(history_df, use_container_width=True, hide_index=True)
            with btn_col:
                if st.button("🗑️ Clear 100m Runs"):
                    st.session_state.saved_runs_100 = []
                    st.rerun()
            st.divider()

        # --- CHARTS (MULTI-RUN OVERLAY) ---
        st.header("📊 Race Analytics")
        if all(f"Velocity {i}0m" in missing_metrics for i in range(1, 10)) or all(f"Time {i}0m" in missing_metrics for i in range(1, 10)):
            st.info("⚠️ Not enough data to plot the charts.")
        else:
            chart_data_list = []
            active_label = "Current Sliders" if len(st.session_state.saved_runs_100) > 0 else "Active Race"
            chart_data_list.append(pd.DataFrame({"Distance (m)": distances, "Velocity (m/s)": velocities, "Acceleration (m/s²)": accelerations, "Run / Athlete": active_label}))
            
            for run in st.session_state.saved_runs_100:
                chart_data_list.append(pd.DataFrame({"Distance (m)": distances, "Velocity (m/s)": run["Velocities"], "Acceleration (m/s²)": run["Accelerations"], "Run / Athlete": run["Run Label"]}))
                
            combined_chart_data = pd.concat(chart_data_list)

            st.subheader("Velocity Curve")
            fig_vel = px.line(combined_chart_data, x="Distance (m)", y="Velocity (m/s)", color="Run / Athlete", markers=True)
            fig_vel.update_xaxes(fixedrange=True, range=[0, 100], showgrid=True, tickvals=distances)
            fig_vel.update_yaxes(fixedrange=True, rangemode="tozero", showgrid=True)
            if len(st.session_state.saved_runs_100) == 0:
                fig_vel.update_traces(line_color="#ff4b4b")
                fig_vel.update_layout(showlegend=False)
            st.plotly_chart(fig_vel, use_container_width=True, config={'displayModeBar': False})

            st.subheader("Acceleration Curve")
            fig_acc = px.line(combined_chart_data, x="Distance (m)", y="Acceleration (m/s²)", color="Run / Athlete", markers=True)
            fig_acc.update_xaxes(fixedrange=True, range=[0, 100], showgrid=True, tickvals=distances)
            fig_acc.update_yaxes(fixedrange=True, showgrid=True)
            fig_acc.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Deceleration Threshold", annotation_position="bottom right")
            if len(st.session_state.saved_runs_100) == 0:
                fig_acc.update_traces(line_color="#00a4fb")
                fig_acc.update_layout(showlegend=False)
            st.plotly_chart(fig_acc, use_container_width=True, config={'displayModeBar': False})

    with tab2:
        st.header("🎯 100m Goal Calculator: Reverse-Engineer a Target Time")
        
        gc_col1, gc_col2 = st.columns(2)
        with gc_col1:
            target_time = st.number_input("Target 100m Finish Time (s)", min_value=9.00, max_value=20.00, value=10.50, step=0.10)
        with gc_col2:
            archetype = st.selectbox("Runner Archetype", ["The Engine (Perfectly balanced speed endurance)", "The Bullet (Incredible start, fades early)", "The Closer (Slow start, huge top speed)"])
        
        if "The Engine" in archetype:
            pacing_ratios = np.array([0.1973, 0.3006, 0.3946, 0.4843, 0.5710, 0.6587, 0.7453, 0.8267, 0.9134])
        elif "The Bullet" in archetype:
            pacing_ratios = np.array([0.1933, 0.2946, 0.3876, 0.4773, 0.5640, 0.6517, 0.7403, 0.8247, 0.9134])
        else: 
            pacing_ratios = np.array([0.2013, 0.3066, 0.4016, 0.4913, 0.5780, 0.6657, 0.7503, 0.8287, 0.9134])

        target_splits = target_time * pacing_ratios
        target_times_full = [0.0] + target_splits.tolist() + [target_time]
        target_distances = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        target_velocities = [0.0]
        for i in range(1, 11):
            dt = target_times_full[i] - target_times_full[i-1]
            target_velocities.append(10 / dt)
            
        st.divider()
        goal_col1, goal_col2 = st.columns([1, 1.5])
        with goal_col1:
            st.subheader("Target Pace Card")
            goal_df = pd.DataFrame({"Mark": ["10m", "20m", "30m", "40m", "50m", "60m", "70m", "80m", "90m", "100m"], "Target Split (s)": target_times_full[1:], "Target Velocity (m/s)": target_velocities[1:]})
            st.dataframe(goal_df.round(2), hide_index=True, use_container_width=True)
            
        with goal_col2:
            st.subheader("Target Velocity Curve")
            goal_chart_data = pd.DataFrame({"Distance (m)": target_distances, "Target Velocity (m/s)": target_velocities})
            fig_goal = px.line(goal_chart_data, x="Distance (m)", y="Target Velocity (m/s)", color_discrete_sequence=["#19c37d"], markers=True)
            fig_goal.update_xaxes(fixedrange=True, range=[0, 100], showgrid=True, tickvals=target_distances)
            fig_goal.update_yaxes(fixedrange=True, rangemode="tozero", showgrid=True)
            st.plotly_chart(fig_goal, use_container_width=True, config={'displayModeBar': False})


# =====================================================================
# 200M DASH EVENT LOGIC
# =====================================================================
elif event_choice == "200m Dash":
    if model_200m is None:
        st.error("Model file '200m_model.pkl' not found. Please run train_and_save_200m.py first.")
        st.stop()
        
    tab1, tab2 = st.tabs(["📊 Predictor & Analysis", "🎯 Goal Calculator"])

    with tab1:
        st.markdown("Adjust the sliders to instantly predict their final 200m time. Calculations evaluate the curve, straightaway, and speed endurance zones.")
        
        st.subheader("Missing Data?")
        missing_metrics_200 = st.multiselect(
            "Select any measurements you do NOT know:",
            [
                "Reaction Time", "Wind", 
                "Time 50m", "Time 100m", "Time 150m",
                "Velocity 50m", "Velocity 100m", "Velocity 150m"
            ],
            placeholder="e.g., Velocity 50m..."
        )
        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.header("Race Conditions")
            reaction_time_200 = st.slider("Reaction Time (s)", min_value=0.100, max_value=0.400, value=0.150, step=0.001, format="%.3f", disabled=("Reaction Time" in missing_metrics_200))
            wind_200 = st.slider("Wind (m/s)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1, format="%.1f", disabled=("Wind" in missing_metrics_200))

        with col2:
            st.header("Split Times (s)")
            t_50 = st.slider("Time 50m", min_value=4.50, max_value=10.00, value=5.95, step=0.01, format="%.2f", disabled=("Time 50m" in missing_metrics_200))
            t_100 = st.slider("Time 100m", min_value=9.00, max_value=20.00, value=10.50, step=0.01, format="%.2f", disabled=("Time 100m" in missing_metrics_200))
            t_150 = st.slider("Time 150m", min_value=13.50, max_value=30.00, value=15.20, step=0.01, format="%.2f", disabled=("Time 150m" in missing_metrics_200))

        with col3:
            st.header("Velocities (m/s)")
            v_50 = st.slider("Velocity 50m", min_value=0.0, max_value=12.0, value=8.40, step=0.01, format="%.2f", disabled=("Velocity 50m" in missing_metrics_200))
            v_100 = st.slider("Velocity 100m", min_value=0.0, max_value=12.0, value=10.98, step=0.01, format="%.2f", disabled=("Velocity 100m" in missing_metrics_200))
            v_150 = st.slider("Velocity 150m", min_value=0.0, max_value=12.0, value=10.63, step=0.01, format="%.2f", disabled=("Velocity 150m" in missing_metrics_200))

        st.divider()

        input_data_200 = {
            "reaction time": [get_val("Reaction Time", reaction_time_200, missing_metrics_200)], 
            "wind": [get_val("Wind", wind_200, missing_metrics_200)],
            "time 50m": [get_val("Time 50m", t_50, missing_metrics_200)], 
            "time 100m": [get_val("Time 100m", t_100, missing_metrics_200)], 
            "time 150m": [get_val("Time 150m", t_150, missing_metrics_200)], 
            "velocity 50m": [get_val("Velocity 50m", v_50, missing_metrics_200)], 
            "velocity 100m": [get_val("Velocity 100m", v_100, missing_metrics_200)], 
            "velocity 150m": [get_val("Velocity 150m", v_150, missing_metrics_200)]
        }

        input_df_200 = pd.DataFrame(input_data_200)
        prediction_200 = model_200m.predict(input_df_200)[0]

        if "Time 150m" in missing_metrics_200:
            v_200 = np.nan
        else:
            final_50m_time = prediction_200 - t_150
            v_200 = 50 / final_50m_time if final_50m_time > 0 else v_150
            
        distances_200 = [0, 50, 100, 150, 200]
        times_200 = [
            0.0, 
            get_val("Time 50m", t_50, missing_metrics_200), 
            get_val("Time 100m", t_100, missing_metrics_200), 
            get_val("Time 150m", t_150, missing_metrics_200), 
            prediction_200
        ]
        velocities_200 = [
            0.0, 
            get_val("Velocity 50m", v_50, missing_metrics_200), 
            get_val("Velocity 100m", v_100, missing_metrics_200), 
            get_val("Velocity 150m", v_150, missing_metrics_200), 
            v_200
        ]
        
        accelerations_200 = [0.0]
        for i in range(1, len(distances_200)):
            dt = times_200[i] - times_200[i-1]
            dv = velocities_200[i] - velocities_200[i-1]
            if pd.isna(dt) or pd.isna(dv) or dt <= 0:
                accelerations_200.append(np.nan)
            else:
                accelerations_200.append(dv / dt)

        # --- PREDICTION METRIC & SAVE RUN CONTROLS ---
        pred_col, save_col = st.columns([1, 1])
        
        with pred_col:
            st.metric(label="Predicted Finish Time", value=f"{prediction_200:.3f} s")
            
        with save_col:
            st.markdown("**Session Comparison**")
            run_label = st.text_input("Run Label / Athlete Name", placeholder="e.g., Heat 1 or Athlete A", label_visibility="collapsed")
            if st.button("💾 Save Run to 200m Session"):
                label_to_save = run_label.strip() if run_label.strip() else f"Run {len(st.session_state.saved_runs_200) + 1}"
                st.session_state.saved_runs_200.append({
                    "Run Label": label_to_save,
                    "Predicted Time (s)": round(prediction_200, 3),
                    "Reaction Time (s)": reaction_time_200 if "Reaction Time" not in missing_metrics_200 else "N/A",
                    "Wind (m/s)": wind_200 if "Wind" not in missing_metrics_200 else "N/A",
                    "Velocities": velocities_200,
                    "Accelerations": accelerations_200
                })
                st.success(f"Saved '{label_to_save}' to session history!")

        st.divider()

        # --- RACE PHASE BREAKDOWN ---
        st.header("⏱️ 200m Race Phase Breakdown")
        if "Time 50m" in missing_metrics_200 or "Time 150m" in missing_metrics_200:
            st.info("⚠️ Please uncheck 'Time 50m' and 'Time 150m' to view the Race Phase Breakdown.")
        else:
            drive_time = t_50
            curve_exit_time = t_150 - t_50
            endurance_time = prediction_200 - t_150
            
            drive_vel = 50 / drive_time if drive_time > 0 else 0
            curve_exit_vel = 100 / curve_exit_time if curve_exit_time > 0 else 0
            endurance_vel = 50 / endurance_time if endurance_time > 0 else 0
            
            ph_col1, ph_col2, ph_col3 = st.columns(3)
            with ph_col1:
                st.metric("Drive Phase (0-50m)", f"{drive_time:.2f} s", f"{drive_vel:.2f} m/s avg", delta_color="off")
            with ph_col2:
                st.metric("Turn Exit & Straight (50-150m)", f"{curve_exit_time:.2f} s", f"{curve_exit_vel:.2f} m/s avg", delta_color="off")
            with ph_col3:
                st.metric("Speed Endurance (150-200m)", f"{endurance_time:.2f} s", f"{endurance_vel:.2f} m/s avg", delta_color="off")

        st.divider()

        # --- SESSION HISTORY TABLE ---
        if len(st.session_state.saved_runs_200) > 0:
            st.header("📋 200m Session History")
            history_df_200 = pd.DataFrame([{ "Run Label": r["Run Label"], "Predicted 200m (s)": r["Predicted Time (s)"], "Reaction (s)": r["Reaction Time (s)"], "Wind (m/s)": r["Wind (m/s)"]} for r in st.session_state.saved_runs_200])
            tbl_col, btn_col = st.columns([3, 1])
            with tbl_col:
                st.dataframe(history_df_200, use_container_width=True, hide_index=True)
            with btn_col:
                if st.button("🗑️ Clear 200m Runs"):
                    st.session_state.saved_runs_200 = []
                    st.rerun()
            st.divider()

        # --- CHARTS (MULTI-RUN OVERLAY) ---
        st.header("📊 200m Race Analytics")
        if all(f"Velocity {i}0m" in missing_metrics_200 for i in [5, 10, 15]) or all(f"Time {i}0m" in missing_metrics_200 for i in [5, 10, 15]):
            st.info("⚠️ Not enough data to plot the charts.")
        else:
            chart_data_list_200 = []
            active_label = "Current Sliders" if len(st.session_state.saved_runs_200) > 0 else "Active Race"
            chart_data_list_200.append(pd.DataFrame({"Distance (m)": distances_200, "Velocity (m/s)": velocities_200, "Acceleration (m/s²)": accelerations_200, "Run / Athlete": active_label}))
            
            for run in st.session_state.saved_runs_200:
                chart_data_list_200.append(pd.DataFrame({"Distance (m)": distances_200, "Velocity (m/s)": run["Velocities"], "Acceleration (m/s²)": run["Accelerations"], "Run / Athlete": run["Run Label"]}))
                
            combined_chart_data_200 = pd.concat(chart_data_list_200)

            st.subheader("Velocity Curve")
            fig_vel_200 = px.line(combined_chart_data_200, x="Distance (m)", y="Velocity (m/s)", color="Run / Athlete", markers=True)
            fig_vel_200.update_xaxes(fixedrange=True, range=[0, 200], showgrid=True, tickvals=distances_200)
            fig_vel_200.update_yaxes(fixedrange=True, rangemode="tozero", showgrid=True)
            if len(st.session_state.saved_runs_200) == 0:
                fig_vel_200.update_traces(line_color="#ff4b4b")
                fig_vel_200.update_layout(showlegend=False)
            st.plotly_chart(fig_vel_200, use_container_width=True, config={'displayModeBar': False})

            st.subheader("Acceleration Curve")
            fig_acc_200 = px.line(combined_chart_data_200, x="Distance (m)", y="Acceleration (m/s²)", color="Run / Athlete", markers=True)
            fig_acc_200.update_xaxes(fixedrange=True, range=[0, 200], showgrid=True, tickvals=distances_200)
            fig_acc_200.update_yaxes(fixedrange=True, showgrid=True)
            fig_acc_200.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Deceleration Threshold", annotation_position="bottom right")
            if len(st.session_state.saved_runs_200) == 0:
                fig_acc_200.update_traces(line_color="#00a4fb")
                fig_acc_200.update_layout(showlegend=False)
            st.plotly_chart(fig_acc_200, use_container_width=True, config={'displayModeBar': False})

    with tab2:
        st.header("🎯 200m Goal Calculator: Reverse-Engineer a Target Time")
        
        gc_col1, gc_col2 = st.columns(2)
        with gc_col1:
            target_time_200 = st.number_input("Target 200m Finish Time (s)", min_value=19.00, max_value=35.00, value=21.50, step=0.10)
        
        # 200m standard pacing approximations
        pacing_ratios_200 = np.array([0.291, 0.516, 0.752])
        target_splits_200 = target_time_200 * pacing_ratios_200
        target_times_full_200 = [0.0] + target_splits_200.tolist() + [target_time_200]
        target_distances_200 = [0, 50, 100, 150, 200]
        
        target_velocities_200 = [0.0]
        for i in range(1, 5):
            dt = target_times_full_200[i] - target_times_full_200[i-1]
            target_velocities_200.append(50 / dt)
            
        st.divider()
        goal_col1_200, goal_col2_200 = st.columns([1, 1.5])
        with goal_col1_200:
            st.subheader("Target Pace Card")
            goal_df_200 = pd.DataFrame({"Mark": ["50m", "100m", "150m", "200m"], "Target Split (s)": target_times_full_200[1:], "Target Velocity (m/s)": target_velocities_200[1:]})
            st.dataframe(goal_df_200.round(2), hide_index=True, use_container_width=True)
            
        with goal_col2_200:
            st.subheader("Target Velocity Curve")
            goal_chart_data_200 = pd.DataFrame({"Distance (m)": target_distances_200, "Target Velocity (m/s)": target_velocities_200})
            fig_goal_200 = px.line(goal_chart_data_200, x="Distance (m)", y="Target Velocity (m/s)", color_discrete_sequence=["#19c37d"], markers=True)
            fig_goal_200.update_xaxes(fixedrange=True, range=[0, 200], showgrid=True, tickvals=target_distances_200)
            fig_goal_200.update_yaxes(fixedrange=True, rangemode="tozero", showgrid=True)
            st.plotly_chart(fig_goal_200, use_container_width=True, config={'displayModeBar': False})