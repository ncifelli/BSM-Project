import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# --- Page Configuration ---
st.set_page_config(layout="wide")

st.title("Portfolio Forecasting")
st.markdown("""
This application runs Monte Carlo simulations for two different bankroll management strategies.
Use the tabs below to switch between a **Static Wager** and a **Compounding Wager** model.
All core betting assumptions (EV, Win Rate, Tiers) remain the same for both models.
""")

# --- Sidebar for User Inputs ---
st.sidebar.header("Simulation Settings")

initial_bankroll = st.sidebar.number_input(
    "Initial Bankroll (Units)",
    min_value=1.0,
    max_value=10000.0,
    value=100.0,
    step=10.0,
    help="The starting bankroll for all simulations."
)

bets_per_day = st.sidebar.number_input(
    "Total Bets Per Day",
    min_value=1,
    max_value=100,
    value=25,
    help="Set the total number of bets you plan to place each day (1-100). These are chosen randomly from a pool of 1000."
)

risk = st.sidebar.slider(
    "Risk Multiplier",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    step=0.01,
    help="Adjusts the risk-reward profile. Higher risk linearly increases EV and decreases Win Rate."
)

st.sidebar.header("Wager Settings")

# --- Wager specific inputs ---
static_total_units_per_day = st.sidebar.number_input(
    "Static Wager: Total Units Per Day",
    min_value=1.0,
    max_value=1000.0,
    value=10.0,
    step=0.5,
    help="For the 'Static Wager' model, this is the fixed total amount you will stake each day."
)

reinvestment_rate = st.sidebar.slider(
    "Compounding: Reinvestment Rate (%)",
    min_value=0.0,
    max_value=100.0,
    value=25.0, # Default to 25%
    step=1.0,
    help="For the 'Compounding Wager' model. Each day, you will wager 10 units PLUS this percentage of any profits you have."
)


# --- Core Simulation Variables ---
num_simulations = 10000
num_days = 90
base_ev = 0.12
base_win_rate = 0.354

# --- Dynamic Calculations based on Inputs ---
ev_multiplier = np.interp(risk, [0.5, 2.0], [0.7, 1.3]) # EV scales from 70% to 130%
wr_multiplier = np.interp(risk, [0.5, 2.0], [1.25, 0.75]) # WR scales from 115% to 85%
effective_ev = base_ev * ev_multiplier
effective_win_rate = base_win_rate * wr_multiplier
reinvestment_decimal = reinvestment_rate / 100.0

# --- Staking Plan Logic (Relative Weights) ---
bet_indices = np.arange(1, bets_per_day + 1)
bet_weights = 1 / np.sqrt(bet_indices)
normalized_weights = bet_weights / np.sum(bet_weights)

# --- Bet Pool Definition ---
bet_pool_definition = [
    {'count': 50, 'ev_mod': 1.2, 'wr_mod': 1.2},
    {'count': 50, 'ev_mod': 1.1, 'wr_mod': 1.1},
    {'count': 100, 'ev_mod': 1.05, 'wr_mod': 1.05},
    {'count': 300, 'ev_mod': 1.0, 'wr_mod': 1.0},
    {'count': 300, 'ev_mod': 0.95, 'wr_mod': 0.95},
    {'count': 100, 'ev_mod': 0.9, 'wr_mod': 0.9},
    {'count': 100, 'ev_mod': 0.8, 'wr_mod': 0.8}
]

# Create the full pool of 1000 bets
bet_pool = []
for bet_type in bet_pool_definition:
    for _ in range(bet_type['count']):
        bet_pool.append({'ev_mod': bet_type['ev_mod'], 'wr_mod': bet_type['wr_mod']})

# Pre-calculate derived parameters for the entire pool
for bet in bet_pool:
    bet['wr'] = np.clip(effective_win_rate * bet['wr_mod'], 0.01, 0.99)
    bet_ev = effective_ev * bet['ev_mod']
    calculated_odds = (bet_ev + 1) / bet['wr']
    if calculated_odds < 1:
        st.error(f"Invalid settings. A bet type calculates odds less than 1 ({calculated_odds:.2f}). Try adjusting risk or base assumptions.", icon="🚨")
        st.stop()
    bet['odds'] = calculated_odds

# For vectorized calculations, create numpy arrays from the pool
pool_wrs = np.array([bet['wr'] for bet in bet_pool])
pool_odds = np.array([bet['odds'] for bet in bet_pool])


# --- Simulation Functions ---

def run_simulation_with_bankruptcy(sims, days, start_bankroll, daily_wager_func, p_wrs, p_odds):
    """
    Generic simulation function that loops day-by-day to handle bankruptcy.
    """
    bankrolls = np.full(sims, start_bankroll)
    cumulative_profit_history = np.zeros((sims, days))
    is_active = np.full(sims, True)
    bankrupt_sims = np.full(sims, False)

    for day in range(days):
        active_indices = np.where(is_active)[0]
        if len(active_indices) == 0: # Stop if all have gone bankrupt
            cumulative_profit_history[:, day:] = cumulative_profit_history[:, day-1, np.newaxis]
            break

        daily_total_wagers = daily_wager_func(bankrolls[active_indices], start_bankroll)
        daily_stakes_matrix = daily_total_wagers[:, np.newaxis] * normalized_weights
        
        num_active_sims = len(active_indices)
        
        # Randomly select 'bets_per_day' bets from the pool for each active simulation
        bet_indices_for_day = np.random.randint(0, len(bet_pool), size=(num_active_sims, bets_per_day))
        
        # Get the corresponding win rates and odds for the selected bets
        sampled_wrs = p_wrs[bet_indices_for_day]
        sampled_odds = p_odds[bet_indices_for_day]
        
        # Simulate outcomes for all bets across all simulations
        random_outcomes = np.random.rand(num_active_sims, bets_per_day)
        is_win = random_outcomes < sampled_wrs
        
        # Calculate profit/loss for each bet
        profit_matrix = np.where(
            is_win,
            daily_stakes_matrix * (sampled_odds - 1),
            -daily_stakes_matrix
        )
        
        # Sum the profits for the day for each simulation
        total_profit_for_day = profit_matrix.sum(axis=1)

        bankrolls[active_indices] += total_profit_for_day
        
        # Update bankruptcy status
        newly_bankrupt = bankrolls < 0
        bankrupt_sims = np.logical_or(bankrupt_sims, newly_bankrupt)
        is_active = bankrolls > 0
        
        cumulative_profit_history[:, day] = bankrolls - start_bankroll

    bankruptcy_rate = np.mean(bankrupt_sims)
    return cumulative_profit_history, bankruptcy_rate

# --- Helper Function for Displaying Results ---
def display_results(cumulative_profit, bankruptcy_rate, stakes_for_histogram, total_wager_for_histogram):
    st.subheader("Simulation Results")
    fig, ax = plt.subplots(figsize=(12, 7))
    days_axis = np.arange(1, num_days + 1)

    median_line = np.percentile(cumulative_profit, 50, axis=0)
    ci_60_upper = np.percentile(cumulative_profit, 80, axis=0)
    ci_60_lower = np.percentile(cumulative_profit, 20, axis=0)
    ci_80_upper = np.percentile(cumulative_profit, 90, axis=0)
    ci_80_lower = np.percentile(cumulative_profit, 10, axis=0)
    ci_95_upper = np.percentile(cumulative_profit, 97.5, axis=0)
    ci_95_lower = np.percentile(cumulative_profit, 2.5, axis=0)

    ax.fill_between(days_axis, ci_95_lower, ci_95_upper, color='#e0e0e0', alpha=0.9, label='95% Range of Outcomes')
    ax.fill_between(days_axis, ci_80_lower, ci_80_upper, color='#bdbdbd', alpha=0.8, label='80% Range of Outcomes')
    ax.fill_between(days_axis, ci_60_lower, ci_60_upper, color='#9e9e9e', alpha=0.7, label='60% Range of Outcomes')
    ax.plot(days_axis, median_line, color='#d32f2f', linestyle='--', linewidth=2, label='Median Outcome')
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_title(f'Future Profit/Loss Distribution ({num_simulations:,} simulations)', fontsize=16)
    ax.set_xlabel('Days', fontsize=12)
    ax.set_ylabel('Cumulative Profit/Loss (Units)', fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(0, num_days)
    ax.set_ylim(np.min(ci_95_lower) * 1.1, np.max(ci_95_upper) * 1.1)
    st.pyplot(fig)

    st.subheader("Key Outcome Metrics at Day 90")
    cols = st.columns(5)
    final_day_outcomes = cumulative_profit[:, -1]
    
    cols[0].metric("Median Outcome", f"{median_line[-1]:.2f} units")
    cols[1].metric("Average Outcome", f"{np.mean(final_day_outcomes):.2f} units")
    cols[2].metric("Probability of Profit", f"{np.mean(final_day_outcomes > 0):.1%}")
    cols[3].metric("5th Percentile", f"{ci_95_lower[-1]:.2f} units")
    cols[4].metric("Bankruptcy Rate", f"{bankruptcy_rate:.1%}", help="Percentage of simulations where the bankroll dropped to 0 or below.")

    st.subheader("Example Daily Staking Distribution")
    stakes_percentage = (stakes_for_histogram / total_wager_for_histogram) * 100
    fig_hist, ax_hist = plt.subplots(figsize=(12, 5))
    ax_hist.hist(stakes_percentage, bins='auto', color='#42a5f5', edgecolor='black')
    ax_hist.set_title('Distribution of Bet Stakes (% of Daily Total)', fontsize=16)
    ax_hist.set_xlabel('% of Daily Units Staked per Bet', fontsize=12)
    ax_hist.set_ylabel('Number of Bets (Frequency)', fontsize=12)
    ax_hist.grid(axis='y', linestyle='--', linewidth=0.5)
    ax_hist.yaxis.set_major_locator(MaxNLocator(integer=True))
    st.pyplot(fig_hist)


# --- Main App Body ---
tab1, tab2 = st.tabs(["Static Wager Simulation", "Compounding Wager Simulation"])

with tab1:
    st.header("📈 Static Wager Model")
    st.write(f"This model simulates wagering a fixed total of **{static_total_units_per_day} units** every day.")
    
    static_wager_func = lambda br, sb: np.full(br.shape, static_total_units_per_day)
    static_results, static_br_rate = run_simulation_with_bankruptcy(num_simulations, num_days, initial_bankroll, static_wager_func, pool_wrs, pool_odds)
    
    static_stakes = normalized_weights * static_total_units_per_day
    display_results(static_results, static_br_rate, static_stakes, static_total_units_per_day)

with tab2:
    st.header("🚀 Compounding Wager Model")
    st.write(f"This model wagers **10 units + {reinvestment_rate:.0f}%** of any accumulated profit each day.")

    compounding_wager_func = lambda br, sb: 10.0 + (reinvestment_decimal * np.maximum(0, br - sb))
    compounding_results, compounding_br_rate = run_simulation_with_bankruptcy(num_simulations, num_days, initial_bankroll, compounding_wager_func, pool_wrs, pool_odds)
    
    day1_stakes = normalized_weights * 10.0
    display_results(compounding_results, compounding_br_rate, day1_stakes, 10.0)