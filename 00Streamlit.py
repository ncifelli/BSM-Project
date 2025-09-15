import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(layout="wide")

st.title("📈 Betting Strategy Monte Carlo Simulation")
st.markdown("""
This application simulates the potential outcomes of a betting strategy over a 90-day period. 
Use the controls on the left to adjust the **risk** and the **number of bets per day** to see how they impact the range of potential profit and loss.
""")

# --- Sidebar for User Inputs ---
st.sidebar.header("Simulation Settings")

# Feature 1: Bets per day (numerical input)
bets_per_day = st.sidebar.number_input(
    "Bets Per Day",
    min_value=1,
    max_value=100,
    value=10,  # Default value
    help="Set the number of bets you plan to place each day (1-100)."
)

# Feature 2: Risk (slider input)
risk = st.sidebar.slider(
    "Risk Multiplier (Moving from left to right increases risk)",
    min_value=0.5,
    max_value=2.0,
    value=1.0,  # Default risk is 1
    step=0.01,
    help="Adjust the risk factor. This directly scales the Expected Value (EV) of each bet."
)

# --- Core Simulation Variables ---
win_rate = 0.354
num_simulations = 10000
num_days = 90
ev = 0.12
# --- Dynamic Calculations based on Inputs ---
# EV is now calculated based on the risk slider
ev = ev * np.sqrt(risk)
win_rate = win_rate * (1/risk)
loss_rate = 1 - win_rate

# Calculate average odds from win rate and EV.
# Added a check to prevent invalid odds.
if win_rate <= 0 or (ev + 1) / win_rate < 1:
    st.error("Invalid combination of Win Rate and EV. Calculated odds are less than 1.")
else:
    decimal_odds = (ev + 1) / win_rate


    # --- Monte Carlo Simulation Logic ---
    # Using st.cache_data to avoid re-running the simulation on every minor widget change
    @st.cache_data
    def run_simulation(sims, days, bets, wr, lr, odds):
        outcomes = np.random.choice([odds - 1, -1],
                                    size=(sims, days * bets),
                                    p=[wr, lr])
        daily_profit = outcomes.reshape(sims, days, bets).sum(axis=2)
        return np.cumsum(daily_profit, axis=1)

    cumulative_profit = run_simulation(num_simulations, num_days, bets_per_day, win_rate, loss_rate, decimal_odds)

    # --- Data Processing for Plotting ---
    median_line = np.percentile(cumulative_profit, 50, axis=0)
    ci_60_upper = np.percentile(cumulative_profit, 80, axis=0)
    ci_60_lower = np.percentile(cumulative_profit, 20, axis=0)
    ci_80_upper = np.percentile(cumulative_profit, 90, axis=0)
    ci_80_lower = np.percentile(cumulative_profit, 10, axis=0)
    ci_95_upper = np.percentile(cumulative_profit, 97.5, axis=0)
    ci_95_lower = np.percentile(cumulative_profit, 2.5, axis=0)

    # --- Plotting ---
    st.subheader("Simulation Results")
    fig, ax = plt.subplots(figsize=(12, 7))
    days_axis = np.arange(1, num_days + 1)

    # Plot confidence intervals
    ax.fill_between(days_axis, ci_95_lower, ci_95_upper, color='#e0e0e0', alpha=0.9, label='95% Confidence Interval')
    ax.fill_between(days_axis, ci_80_lower, ci_80_upper, color='#bdbdbd', alpha=0.8, label='80% Confidence Interval')
    ax.fill_between(days_axis, ci_60_lower, ci_60_upper, color='#9e9e9e', alpha=0.7, label='60% Confidence Interval')
    
    # Plot the median line
    ax.plot(days_axis, median_line, color='#d32f2f', linestyle='--', linewidth=2, label='Median Outcome')
    
    # Add a horizontal line at 0
    ax.axhline(0, color='black', linestyle='-', linewidth=1)

    # Aesthetics and labels
    ax.set_title(f'Future Profit/Loss Distribution ({num_simulations:,} simulations)', fontsize=16)
    ax.set_xlabel('Days', fontsize=12)
    ax.set_ylabel('Cumulative Profit/Loss (Units)', fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(0, num_days)

    st.pyplot(fig)
