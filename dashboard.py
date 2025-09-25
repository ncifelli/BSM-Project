import streamlit as st
import pandas as pd
import altair as alt
from supabase import create_client
import os
from dotenv import load_dotenv
from pathlib import Path
from streamlit_autorefresh import st_autorefresh
from datetime import date, datetime, time

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Betting Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 2. Auto-Refresh Component ---
REFRESH_INTERVAL_SECONDS = 60
st_autorefresh(interval=REFRESH_INTERVAL_SECONDS * 1000, key="data_refresher")

# --- 3. Supabase Connection ---
@st.cache_resource
def init_supabase_client():
    """Initializes and returns a Supabase client."""
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    except (KeyError, st.errors.StreamlitSecretNotFoundError):
        try:
            repo_root = Path(__file__).parent.parent
            load_dotenv(repo_root / ".env")
        except Exception:
            load_dotenv()
        url = os.getenv("SUPABASE_URL", 'https://vnbiigaqhwjbstzzbvjb.supabase.co')
        key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        st.error("Supabase URL and Key are not set. Please add them to your .env file or Streamlit secrets.")
        st.stop()
    return create_client(url, key)

supabase = init_supabase_client()

# --- 4. Session State Initialization ---
if 'master_df' not in st.session_state:
    st.session_state.master_df = None
if 'last_fetch_time' not in st.session_state:
    st.session_state.last_fetch_time = None

# --- 5. Data Loading and Processing Functions ---
def load_data_incrementally():
    """
    Loads data incrementally, handling datetime conversions correctly.
    """
    select_columns = 'market, placed_at, actual_return, bet_amount, sportsbook, sport'

    if st.session_state.last_fetch_time is None:
        st.info("Performing initial full data load...")
        response = supabase.table('betting_history').select(select_columns).order('placed_at', desc=True).execute()
        df = pd.DataFrame(response.data)
        if not df.empty:
            df['placed_at'] = pd.to_datetime(df['placed_at'], format='ISO8601')
        st.session_state.master_df = df
    else:
        last_time_str = st.session_state.last_fetch_time.isoformat()
        response = supabase.table('betting_history').select(select_columns).gt('placed_at', last_time_str).execute()
        new_df = pd.DataFrame(response.data)
        
        if not new_df.empty:
            st.toast(f"Found {len(new_df)} new record(s)!")
            new_df['placed_at'] = pd.to_datetime(new_df['placed_at'], format='ISO8601')
            st.session_state.master_df = pd.concat([new_df, st.session_state.master_df], ignore_index=True)

    if st.session_state.master_df is not None and not st.session_state.master_df.empty:
        st.session_state.last_fetch_time = st.session_state.master_df['placed_at'].max()

    return st.session_state.master_df

def process_data(df, cutoff_datetime):
    """
    Performs data cleaning and adds a dynamic 'period' column based on the user-selected cutoff.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df_processed = df.copy()

    for col in ['sport', 'sportsbook', 'market']:
        if col in df_processed.columns and not df_processed[col].empty and isinstance(df_processed[col].dropna().iloc[0], dict):
            df_processed[col] = df_processed[col].apply(lambda x: x.get('name') if isinstance(x, dict) else x)

    numeric_cols = ['actual_return', 'bet_amount']
    for col in numeric_cols:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    df_processed.dropna(subset=numeric_cols, inplace=True)
    
    # --- Dynamic period assignment ---
    df_processed['period'] = 'Post-Cutoff'
    df_processed.loc[df_processed['placed_at'] < cutoff_datetime, 'period'] = 'Pre-Cutoff'

    sport_counts = df_processed['sport'].value_counts()
    baseball_variants = [s for s in sport_counts.index if s and 'baseball' in s.lower()]

    if len(baseball_variants) == 2:
        major, other = (baseball_variants[0], baseball_variants[1]) if sport_counts[baseball_variants[0]] > sport_counts[baseball_variants[1]] else (baseball_variants[1], baseball_variants[0])
        df_processed['sport'] = df_processed['sport'].replace({major: 'Baseball (MLB)', other: 'Baseball (Other)'})

    return df_processed

# --- 6. Main Application Flow ---
raw_df = load_data_incrementally()

# --- 7. Sidebar and UI Filters ---
st.sidebar.header("Dashboard Filters")

if raw_df is None or raw_df.empty:
    st.sidebar.warning("No data available to display filters.")
else:
    # --- Dynamic Cutoff Date Selector ---
    st.sidebar.subheader("Analysis Cutoff")
    today = date.today()
    start_date = date(2025, 8, 8)
    
    selected_date = st.sidebar.date_input(
        "Select Cutoff Date:",
        value=date(2025, 9, 18),
        min_value=start_date,
        max_value=today,
        help="Performance will be compared before and after 12:00 PM (UTC) on this date."
    )
    
    # --- Create a timezone-aware timestamp for accurate comparison ---
    cutoff_datetime = pd.Timestamp(datetime.combine(selected_date, time(12, 0)), tz='UTC')

    # Process data with the selected cutoff
    master_df = process_data(raw_df, cutoff_datetime)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Display Options")
    
    min_bet_percentage = st.sidebar.slider(
        "Minimum Bet % for Market Chart:",
        min_value=0.0, max_value=5.0, value=0.5, step=0.1,
        format='%.1f%%', key='min_bet_percentage'
    )
    
    metric_selection = st.sidebar.radio(
        "Select Primary Metric:",
        ('Return on Investment (ROI)', 'Number of Bets'),
        key='metric_selection'
    )

    # --- Updated labels for the view selector ---
    comparison_view = st.sidebar.selectbox(
        "Comparison View:",
        ('Comparison View', 'Pre-Cutoff Only', 'Post-Cutoff Only'),
        key='comparison_view'
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Filters")

    all_sportsbooks = sorted(master_df['sportsbook'].dropna().unique())
    default_books = ['DraftKings', 'BetMGM', 'Pinnacle', 'Caesars', 'Fanatics', 'ESPN BET']
    existing_default_books = [book for book in default_books if book in all_sportsbooks]
    
    if st.sidebar.checkbox("Select All Sportsbooks", value=False):
        default_selection = all_sportsbooks
    else:
        default_selection = existing_default_books
    
    selected_sportsbooks = st.sidebar.multiselect(
        "Filter by Sportsbook:",
        all_sportsbooks,
        default=default_selection
    )

    all_sports = sorted(master_df['sport'].dropna().unique())
    selected_sports = st.sidebar.multiselect(
        "Filter by Sport:",
        all_sports,
        default=all_sports
    )

    all_markets = sorted(master_df['market'].dropna().unique())
    if st.sidebar.checkbox("Select All Markets", value=True):
        default_mkt_selection = all_markets
    else:
        default_mkt_selection = []

    selected_markets = st.sidebar.multiselect(
        "Filter by Market:",
        all_markets,
        default=default_mkt_selection
    )

    filtered_df = master_df[
        master_df['sportsbook'].isin(selected_sportsbooks) &
        master_df['sport'].isin(selected_sports) &
        master_df['market'].isin(selected_markets)
    ]

    # --- Filter based on the selected period view ---
    if comparison_view == 'Pre-Cutoff Only':
        filtered_df = filtered_df[filtered_df['period'] == 'Pre-Cutoff']
    elif comparison_view == 'Post-Cutoff Only':
        filtered_df = filtered_df[filtered_df['period'] == 'Post-Cutoff']

# --- 8. Main Panel: Title and KPIs ---
st.title("📈 Betting Performance Analysis")
st.write(f"Dashboard automatically checks for new data every {REFRESH_INTERVAL_SECONDS} seconds.")

if 'filtered_df' not in locals() or filtered_df.empty:
    st.warning("No data matches the current filter settings.")
else:
    total_wagered = filtered_df['bet_amount'].sum()
    total_returned = filtered_df['actual_return'].sum()
    profit = total_returned - total_wagered
    roi = (profit / total_wagered) if total_wagered > 0 else 0
    num_bets = len(filtered_df)

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(label="Total Bets Placed", value=f"{num_bets:,}")
    kpi2.metric(label="Total Wagered", value=f"${total_wagered:,.2f}")
    kpi3.metric(label="Overall ROI", value=f"{roi:.2%}")

    # --- 9. Charting Logic ---
    st.markdown("---")
    st.header("Performance Visualizations")

    def prepare_chart_data(df, group_by_col):
        group_fields = [group_by_col]
        if comparison_view == 'Comparison View':
            group_fields.append('period')

        if metric_selection == 'Number of Bets':
            chart_df = df.groupby(group_fields).size().reset_index(name='Number of Bets')
        else: # ROI
            def calculate_roi(group):
                wagered = group['bet_amount'].sum()
                returned = group['actual_return'].sum()
                return (returned - wagered) / wagered if wagered > 0 else 0
            
            roi_df = df.groupby(group_fields).apply(calculate_roi).reset_index(name='Return on Investment (ROI)')
            count_df = df.groupby(group_fields).size().reset_index(name='Number of Bets')
            total_count_df = df.groupby(group_by_col).size().reset_index(name='Total Bets')

            chart_df = pd.merge(roi_df, count_df, on=group_fields)
            chart_df = pd.merge(chart_df, total_count_df, on=group_by_col)

        return chart_df

    def create_bar_chart(data, group_by_col):
        y_axis_format = '%' if metric_selection == 'Return on Investment (ROI)' else ''
        sort_order = alt.SortField(field="Total Bets" if metric_selection == 'Return on Investment (ROI)' else metric_selection, order="descending")

        tooltip = [
            alt.Tooltip(group_by_col, title=group_by_col.replace('_', ' ').title()),
            alt.Tooltip(metric_selection, format=y_axis_format),
            alt.Tooltip('Number of Bets', format=',')
        ]

        encode_params = {
            'x': alt.X(f'{group_by_col}:N', title=group_by_col.title(), sort=sort_order),
            'y': alt.Y(f'{metric_selection}:Q', title=metric_selection, axis=alt.Axis(format=y_axis_format)),
            'tooltip': tooltip
        }
        
        # --- Use 'period' for comparison encoding ---
        if comparison_view == 'Comparison View':
            tooltip.append(alt.Tooltip('period', title='Period'))
            encode_params['color'] = alt.Color('period:N', title='Period')
            encode_params['xOffset'] = alt.XOffset('period:N')
        else:
            encode_params['color'] = alt.value('#FF6347')

        chart = alt.Chart(data).mark_bar().encode(
            **encode_params
        ).properties(
            title=f'{metric_selection} by {group_by_col.title()}'
        ).interactive()

        return chart

    # By Sportsbook
    sportsbook_data = prepare_chart_data(filtered_df, 'sportsbook')
    if not sportsbook_data.empty:
        st.altair_chart(create_bar_chart(sportsbook_data, 'sportsbook'), use_container_width=True)
    else:
        st.info("No data for Sportsbook analysis with current filters.")
        
    # By Sport
    sport_data = prepare_chart_data(filtered_df, 'sport')
    if not sport_data.empty:
        st.altair_chart(create_bar_chart(sport_data, 'sport'), use_container_width=True)
    else:
        st.info("No data for Sport analysis with current filters.")

    # By Market
    min_bets_threshold = (min_bet_percentage / 100) * len(filtered_df)
    significant_markets = filtered_df['market'].value_counts()
    significant_markets = significant_markets[significant_markets > min_bets_threshold].index.tolist()
    
    market_df_for_charting = filtered_df[filtered_df['market'].isin(significant_markets)]
    
    if not market_df_for_charting.empty:
        market_data = prepare_chart_data(market_df_for_charting, 'market')
        st.altair_chart(create_bar_chart(market_data, 'market'), use_container_width=True)
    else:
        st.info(f"No significant market data to display (markets with >{min_bets_threshold:.1f} bets).")