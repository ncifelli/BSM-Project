import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import requests
import datetime
import numpy as np
from supabase import create_client
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---
load_dotenv()

OPTIC_ODDS_API_URL = "https://api.opticodds.com/api/v3/grader/odds"
CUTOFF_HOUR_UTC = 10 
DATA_START_CUTOFF = "2025-12-12T10:00:00Z" 

st.set_page_config(page_title="Live Betting Portfolio", page_icon="📈", layout="wide")

# --- 2. SESSION STATE ---
if 'local_updates' not in st.session_state:
    st.session_state['local_updates'] = {}

# --- 3. CONNECTIONS ---
@st.cache_resource
def init_supabase():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        st.error("❌ Missing Supabase credentials.")
        return None
    return create_client(url, key)

def get_api_key():
    return os.environ.get("OPTIC_ODDS_API_KEY")

# --- 4. DATA FETCHING ---
def fetch_data(supabase):
    try:
        response = supabase.table('portfolio_bets') \
            .select('*') \
            .eq('portfolio_type', 'american') \
            .gt('placed_at', DATA_START_CUTOFF) \
            .execute()
        
        df = pd.DataFrame(response.data)
        
        if df.empty: return df

        if 'result' not in df.columns:
            df['result'] = 'Pending'
        
        required_cols = ['odds', 'weight', 'sport', 'sportsbook', 'selection_name']
        for col in required_cols:
            if col not in df.columns:
                df[col] = None

        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# --- 5. DATA PROCESSING ---
def apply_local_grades(df):
    if df.empty or not st.session_state['local_updates']: return df
    def get_current_status(row):
        if row['id'] in st.session_state['local_updates']:
            return st.session_state['local_updates'][row['id']]
        return row['result']
    df['result'] = df.apply(get_current_status, axis=1)
    return df

def process_dataframe(df):
    if df.empty: return df

    df = apply_local_grades(df)

    if 'placed_at' in df.columns:
        df['placed_at'] = pd.to_datetime(df['placed_at'], format='mixed', utc=True)
        df = df.sort_values('placed_at')

    for col in ['odds', 'fair_odds', 'weight']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    def calculate_row_pnl(row):
        res = str(row.get('result', '')).lower()
        odds = row['odds']
        stake = row['weight']
        if res in ['won', 'win']: return stake * (odds - 1.0)
        if res in ['lost', 'loss', 'lose']: return -stake
        if res == 'half won': return (stake * (odds - 1.0)) * 0.5
        if res == 'half lost': return -0.5 * stake
        return 0.0 

    df['pnl'] = df.apply(calculate_row_pnl, axis=1)

    def get_sports_day(dt):
        if pd.isnull(dt): return None
        if dt.hour < CUTOFF_HOUR_UTC:
            return (dt - datetime.timedelta(days=1)).date()
        return dt.date()

    df['sports_day'] = df['placed_at'].apply(get_sports_day)
    return df

# --- 6. ADVANCED METRICS & MONTE CARLO ---
def calculate_max_drawdown(df):
    """Calculates Max Drawdown from a PnL Series."""
    if df.empty: return 0.0
    
    cumulative = df['pnl'].cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    
    # If the curve is all positive, drawdown is 0. If negative, it's the min value.
    min_drawdown = drawdown.min()
    return min_drawdown

def calculate_profit_factor(df):
    """Calculates Gross Win / Gross Loss."""
    if df.empty: return 0.0
    wins = df[df['pnl'] > 0]['pnl'].sum()
    losses = abs(df[df['pnl'] < 0]['pnl'].sum())
    
    if losses == 0: return float('inf') if wins > 0 else 0.0
    return wins / losses

def run_monte_carlo(df, num_simulations=1000):
    if df.empty:
        return df, {}
    
    df = df.sort_values('placed_at').copy()
    
    # Prep Arrays
    weights = df['weight'].fillna(0).to_numpy()
    
    odds_series = df['odds'].fillna(1.01)
    if 'fair_odds' in df.columns:
        fair_odds_series = df['fair_odds'].fillna(odds_series)
    else:
        fair_odds_series = odds_series

    odds = odds_series.to_numpy()
    fair_odds = fair_odds_series.to_numpy()
    fair_odds = np.where(fair_odds <= 1.0, 1.01, fair_odds) 
    
    # Simulation Logic
    win_probs = 1.0 / fair_odds
    potential_profit = weights * (odds - 1.0)
    potential_loss = -weights
    
    m = len(weights) # Number of bets
    
    # 1000 Simulations x M bets
    random_matrix = np.random.rand(num_simulations, m)
    wins_matrix = random_matrix < win_probs
    sim_pnl_matrix = np.where(wins_matrix, potential_profit, potential_loss)
    
    cumulative_paths = np.cumsum(sim_pnl_matrix, axis=1)
    
    # Chart Data
    df['ci_95_upper'] = np.quantile(cumulative_paths, 0.975, axis=0)
    df['ci_95_lower'] = np.quantile(cumulative_paths, 0.025, axis=0)
    df['ci_60_upper'] = np.quantile(cumulative_paths, 0.80, axis=0)
    df['ci_60_lower'] = np.quantile(cumulative_paths, 0.20, axis=0)
    
    # --- METRICS CALCULATION ---
    # We analyze the distribution of the FINAL outcomes
    final_outcomes = cumulative_paths[:, -1]
    
    expected_profit = np.mean(final_outcomes)
    std_dev_total = np.std(final_outcomes)
    prob_profit = np.mean(final_outcomes > 0)
    
    # 1. Z-Score (Total Mean / Total StdDev)
    z_score = expected_profit / std_dev_total if std_dev_total != 0 else 0.0
    
    # 2. Sharpe (Per Bet)
    # Approximation: Z-Score / sqrt(N)
    sharpe_per_bet = z_score / np.sqrt(m) if m > 0 else 0.0
    
    # 3. Annualized Sharpe
    # Logic: Calculate bets per day, extrapolate to year
    time_span = (df['placed_at'].max() - df['placed_at'].min()).total_seconds()
    if time_span > 0:
        days = time_span / 86400
        # If less than 1 day, normalize to 1 day to avoid explosion
        days = max(days, 1.0)
        bets_per_year = (m / days) * 365
        sharpe_annual = sharpe_per_bet * np.sqrt(bets_per_year)
    else:
        # Fallback if only 1 bet or instant time
        sharpe_annual = 0.0
        
    metrics = {
        "expected_profit": expected_profit,
        "std_dev_total": std_dev_total,
        "prob_profit": prob_profit,
        "z_score": z_score,
        "sharpe_per_bet": sharpe_per_bet,
        "sharpe_annual": sharpe_annual
    }
    
    return df, metrics

# --- 7. DAILY LIMITER ---
def limit_bets_per_day(df, limit):
    if df.empty or limit == "All": return df
    
    limit = int(limit)
    def sampler(group):
        if len(group) > limit:
            return group.sample(n=limit) 
        return group

    limited_df = df.groupby('sports_day', group_keys=False).apply(sampler)
    return limited_df.sort_values('placed_at')

# --- 8. API GRADING ---
def grade_ungraded_bets(df, api_key):
    df = apply_local_grades(df)
    mask = df['result'].isna() | (df['result'].astype(str).str.lower().isin(['pending', '', 'nan', 'none']))
    pending_bets = df[mask]
    
    if pending_bets.empty: return 0

    updates_count = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(pending_bets)
    for i, (index, row) in enumerate(pending_bets.iterrows()):
        progress_bar.progress((i + 1) / total)
        status_text.text(f"Checking: {row.get('selection_name', 'Unknown')}")
        
        fixture_id = str(row.get('selection_id')).split('||')[0] if '||' in str(row.get('selection_id')) else row.get('fixture_id')
        params = {
            "sport": row.get('sport'), "fixture_id": fixture_id,
            "market": row.get('market'), "name": row.get('selection_name'),
        }
        headers = {"X-Api-Key": api_key, "Accept": "application/json"}
        
        try:
            response = requests.get(OPTIC_ODDS_API_URL, headers=headers, params=params, timeout=4)
            if response.status_code == 200:
                new_grade = response.json().get('data', {}).get('result')
                if new_grade and new_grade not in ['Pending', None]:
                    st.session_state['local_updates'][row['id']] = new_grade
                    updates_count += 1
        except: continue

    status_text.empty()
    progress_bar.empty()
    return updates_count

# --- 9. RENDERER ---
def render_metrics_and_chart(df, title):
    if df.empty:
        st.info(f"No bets found for {title}.")
        return

    # Realized Stats
    total_pnl = df['pnl'].sum()
    total_staked = df['weight'].sum()
    roi = (total_pnl / total_staked * 100) if total_staked else 0.0
    settled = df[~df['result'].astype(str).str.lower().isin(['pending', '', 'nan', 'none'])]
    win_count = settled[settled['pnl'] > 0].shape[0]
    win_rate = (win_count / len(settled) * 100) if not settled.empty else 0.0
    
    # New Realized Metrics
    profit_factor = calculate_profit_factor(settled)
    max_dd = calculate_max_drawdown(df)

    # Monte Carlo Stats
    df_chart = df.sort_values('placed_at').copy()
    df_chart, mc_metrics = run_monte_carlo(df_chart, num_simulations=1000)
    df_chart['cumulative_pnl'] = df_chart['pnl'].cumsum()
    
    # Expected ROI (Expected Profit / Volume)
    exp_roi = (mc_metrics['expected_profit'] / total_staked * 100) if total_staked else 0.0

    # --- ROW 1: REALIZED ---
    st.markdown("#### 🏁 Realized Performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Net PnL", f"${total_pnl:,.2f}")
    c2.metric("ROI", f"{roi:.2f}%")
    c3.metric("Max Drawdown", f"${max_dd:,.2f}", help="Maximum peak-to-valley loss")
    c4.metric("Profit Factor", f"{profit_factor:.2f}", help="Gross Wins / Gross Losses")
    c5.metric("Win Rate", f"{win_rate:.1f}%")
    
    # --- ROW 2: EXPECTED (MONTE CARLO) ---
    st.markdown("#### 🔮 Monte Carlo Expectations (Fair Odds)")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    
    m1.metric("Expected Profit", f"${mc_metrics['expected_profit']:,.2f}")
    m2.metric("Expected ROI", f"{exp_roi:.2f}%")
    m3.metric("Prob. Profit", f"{mc_metrics['prob_profit']*100:.1f}%")
    m4.metric("Sharpe (Per Bet)", f"{mc_metrics['sharpe_per_bet']:.3f}")
    m5.metric("Sharpe (Annual)", f"{mc_metrics['sharpe_annual']:.2f}")
    m6.metric("Z-Score", f"{mc_metrics['z_score']:.2f}", help="Total Mean / Total StdDev")

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_chart['placed_at'], y=df_chart['ci_95_upper'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=df_chart['placed_at'], y=df_chart['ci_95_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(128, 128, 128, 0.2)', name='95% CI'))
    fig.add_trace(go.Scatter(x=df_chart['placed_at'], y=df_chart['ci_60_upper'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=df_chart['placed_at'], y=df_chart['ci_60_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(128, 128, 128, 0.2)', name='60% CI'))
    fig.add_trace(go.Scatter(x=df_chart['placed_at'], y=df_chart['cumulative_pnl'], mode='lines+markers', line=dict(color='royalblue', width=2), name='Actual PnL'))
    fig.update_layout(title=f"Performance ({title})", xaxis_title="Time", yaxis_title="Profit ($)", hovermode="x unified", legend=dict(orientation="h", y=1.02, x=1))
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3)
    st.plotly_chart(fig, use_container_width=True)
    
    # Live Scores Widget
    st.info("📢 **Live Scores go here**")

# --- 10. MAIN ---
def main():
    supabase = init_supabase()
    api_key = get_api_key()
    if not supabase: st.stop()

    st.sidebar.header("Controls")
    raw_df = fetch_data(supabase)
    
    if st.sidebar.button("🔄 Check New Bets (Session Only)", type="primary"):
        with st.spinner("Checking API..."):
            updates = grade_ungraded_bets(raw_df, api_key)
            if updates > 0: st.success(f"Updated {updates} bets!")
            st.rerun()

    df = process_dataframe(raw_df)

    if df.empty:
        st.warning(f"No bets found after {DATA_START_CUTOFF}")
        st.stop()

    # Filters
    st.sidebar.subheader("Filters")
    limit_val = st.sidebar.select_slider("Max Bets Per Day (Random Sample)", options=list(range(1, 21)) + ["All"], value="All")
    
    sports = st.sidebar.multiselect("Sport", sorted(df['sport'].dropna().unique()), default=sorted(df['sport'].dropna().unique()))
    books = st.sidebar.multiselect("Sportsbook", sorted(df['sportsbook'].dropna().unique()), default=sorted(df['sportsbook'].dropna().unique()))
    
    filtered_df = df[(df['sport'].isin(sports)) & (df['sportsbook'].isin(books))]
    filtered_df = limit_bets_per_day(filtered_df, limit_val)

    # Dashboard
    st.title("📊 Live Betting Dashboard")
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    current_sports_day = now_utc.date()
    if now_utc.hour < CUTOFF_HOUR_UTC: current_sports_day -= datetime.timedelta(days=1)

    tab1, tab2, tab3 = st.tabs(["📆 Current Day", "🗓 Last 7 Days", "♾️ All Time"])

    with tab1:
        render_metrics_and_chart(filtered_df[filtered_df['sports_day'] == current_sports_day], "Current Sports Day")
    with tab2:
        start = current_sports_day - datetime.timedelta(days=6)
        render_metrics_and_chart(filtered_df[filtered_df['sports_day'] >= start], "Last 7 Days")
    with tab3:
        render_metrics_and_chart(filtered_df, "All Time")

    # Footer
    st.markdown("---")
    st.subheader("⏳ Open / Unsettled Bets")
    unsettled = filtered_df[filtered_df['result'].astype(str).str.lower().isin(['pending', '', 'nan', 'none'])]
    if not unsettled.empty:
        cols = ['placed_at', 'sport', 'selection_name', 'market', 'sportsbook', 'odds', 'fair_odds', 'weight']
        show = [c for c in cols if c in unsettled.columns]
        st.dataframe(unsettled[show].style.format({"odds": "{:.2f}", "fair_odds": "{:.2f}", "weight": "{:.2f}"}), use_container_width=True, hide_index=True)
    else: st.success("No open bets.")

    st.markdown("---")
    st.warning("📰 **Relevant news go here**")

if __name__ == "__main__":
    main()