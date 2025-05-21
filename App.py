import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Black-Scholes-Merton (BSM) Functions ---

def d1_calc(S0, K, r, s, T):
    """Calculates the d1 term of the Black-Scholes-Merton model."""
    # Ensure T is not zero or negative to avoid math errors
    if T <= 0:
        return np.nan # Return NaN for invalid time
    return (np.log(S0 / K) + (r + 0.5 * s**2) * T) / (s * np.sqrt(T))

def d2_calc(d1, s, T):
    """Calculates the d2 term of the Black-Scholes-Merton model."""
    if T <= 0:
        return np.nan # Return NaN for invalid time
    return d1 - s * np.sqrt(T)

def call_price_bsm(S0, K, r, s, T):
    """Calculates the European Call Option price using BSM."""
    d1 = d1_calc(S0, K, r, s, T)
    d2 = d2_calc(d1, s, T)
    # Handle cases where d1 or d2 might be NaN due to invalid T
    if np.isnan(d1) or np.isnan(d2):
        return np.nan
    call = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call

def put_price_bsm(S0, K, r, s, T):
    """Calculates the European Put Option price using BSM."""
    d1 = d1_calc(S0, K, r, s, T)
    d2 = d2_calc(d1, s, T)
    # Handle cases where d1 or d2 might be NaN due to invalid T
    if np.isnan(d1) or np.isnan(d2):
        return np.nan
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return put

# --- BSM Greeks Functions ---

def delta_call_bsm(S0, K, r, s, T):
    """Calculates the Delta for a European Call Option."""
    d1 = d1_calc(S0, K, r, s, T)
    if np.isnan(d1): return np.nan
    return norm.cdf(d1)

def delta_put_bsm(S0, K, r, s, T):
    """Calculates the Delta for a European Put Option."""
    d1 = d1_calc(S0, K, r, s, T)
    if np.isnan(d1): return np.nan
    return norm.cdf(d1) - 1

def gamma_bsm(S0, K, r, s, T):
    """Calculates the Gamma for both European Call and Put Options."""
    d1 = d1_calc(S0, K, r, s, T)
    if np.isnan(d1) or s == 0 or T == 0: return np.nan # Avoid division by zero
    return norm.pdf(d1) / (S0 * s * np.sqrt(T))

def vega_bsm(S0, K, r, s, T):
    """Calculates the Vega for both European Call and Put Options."""
    d1 = d1_calc(S0, K, r, s, T)
    if np.isnan(d1) or T == 0: return np.nan # Avoid sqrt(0)
    # Vega is often quoted per 1% change in volatility, so we divide by 100
    return (S0 * np.sqrt(T) * norm.pdf(d1)) / 100

def theta_call_bsm(S0, K, r, s, T):
    """Calculates the Theta for a European Call Option (per year)."""
    d1 = d1_calc(S0, K, r, s, T)
    d2 = d2_calc(d1, s, T)
    if np.isnan(d1) or np.isnan(d2) or T == 0: return np.nan
    term1 = -(S0 * norm.pdf(d1) * s) / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
    return term1 - term2

def theta_put_bsm(S0, K, r, s, T):
    """Calculates the Theta for a European Put Option (per year)."""
    d1 = d1_calc(S0, K, r, s, T)
    d2 = d2_calc(d1, s, T)
    if np.isnan(d1) or np.isnan(d2) or T == 0: return np.nan
    term1 = -(S0 * norm.pdf(d1) * s) / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
    return term1 + term2

def rho_call_bsm(S0, K, r, s, T):
    """Calculates the Rho for a European Call Option."""
    d2 = d2_calc(d1_calc(S0, K, r, s, T), s, T)
    if np.isnan(d2): return np.nan
    return K * T * np.exp(-r * T) * norm.cdf(d2)

def rho_put_bsm(S0, K, r, s, T):
    """Calculates the Rho for a European Put Option."""
    d2 = d2_calc(d1_calc(S0, K, r, s, T), s, T)
    if np.isnan(d2): return np.nan
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)

# --- Streamlit Application Layout ---

st.set_page_config(layout="wide") # Use wide layout for better plot display
st.title("Basic Option Pricing With BSM & 3D Visualizations")

st.write("---") # Adds a horizontal line for visual separation

# Sidebar for inputs
st.sidebar.header("Option Parameters")
S0 = st.sidebar.number_input("Asset Price (S0)", min_value=0.01, value=100.0, step=0.1)
K = st.sidebar.number_input("Strike Price (K)", min_value=0.01, value=100.0, step=0.1)
r = st.sidebar.number_input("Interest Rate (r, e.g., 0.05 for 5%)", min_value=0.0, value=0.05, step=0.001, format="%.4f")
s = st.sidebar.number_input("Volatility (σ, e.g., 0.20 for 20%)", min_value=0.001, value=0.20, step=0.001, format="%.4f")
T = st.sidebar.number_input("Time Until Expiration (T, in years)", min_value=0.001, value=1.0, step=0.01)

st.write("## Option Prices")

# Ensure T is not zero or negative to avoid math errors for price calculation
if T <= 0:
    st.error("Time Until Expiration (T) must be greater than 0 for price calculation.")
    # Initialize Call and Put to NaN if T is invalid
    Call = np.nan
    Put = np.nan
else:
    # Calculate Call and Put prices for the current inputs
    Call = call_price_bsm(S0, K, r, s, T)
    Put = put_price_bsm(S0, K, r, s, T)

    # Display the results using st.metric
    col_call_price, col_put_price = st.columns(2)
    with col_call_price:
        if not np.isnan(Call):
            st.metric(label="Call Option Price", value=f"${Call:.2f}")
        else:
            st.metric(label="Call Option Price", value="N/A")
    with col_put_price:
        if not np.isnan(Put):
            st.metric(label="Put Option Price", value=f"${Put:.2f}")
        else:
            st.metric(label="Put Option Price", value="N/A")

st.write("---")

# --- Option Greeks Section ---
st.write("## Option Greeks")

if T <= 0:
    st.error("Time Until Expiration (T) must be greater than 0 to calculate Greeks.")
    # Display N/A for all Greeks if T is invalid
    greek_values = {
        "Delta Call": np.nan, "Delta Put": np.nan,
        "Gamma": np.nan,
        "Vega": np.nan,
        "Theta Call (per year)": np.nan, "Theta Put (per year)": np.nan,
        "Rho Call": np.nan, "Rho Put": np.nan
    }
else:
    # Calculate Greeks
    delta_c = delta_call_bsm(S0, K, r, s, T)
    delta_p = delta_put_bsm(S0, K, r, s, T)
    gamma = gamma_bsm(S0, K, r, s, T)
    vega = vega_bsm(S0, K, r, s, T) # This is already per 1% change
    theta_c = theta_call_bsm(S0, K, r, s, T)
    theta_p = theta_put_bsm(S0, K, r, s, T)
    rho_c = rho_call_bsm(S0, K, r, s, T)
    rho_p = rho_put_bsm(S0, K, r, s, T)

    greek_values = {
        "Delta Call": delta_c,
        "Delta Put": delta_p,
        "Gamma": gamma,
        "Vega": vega, # This is per 1% change
        "Theta Call (per year)": theta_c,
        "Theta Put (per year)": theta_p,
        "Rho Call": rho_c,
        "Rho Put": rho_p
    }

# Display Greeks in columns
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Delta Call", value=f"{greek_values['Delta Call']:.4f}" if not np.isnan(greek_values['Delta Call']) else "N/A")
    st.metric(label="Gamma", value=f"{greek_values['Gamma']:.4f}" if not np.isnan(greek_values['Gamma']) else "N/A")
    st.metric(label="Rho Call", value=f"{greek_values['Rho Call']:.4f}" if not np.isnan(greek_values['Rho Call']) else "N/A")

with col2:
    st.metric(label="Delta Put", value=f"{greek_values['Delta Put']:.4f}" if not np.isnan(greek_values['Delta Put']) else "N/A")
    st.metric(label="Vega (per 1% σ)", value=f"{greek_values['Vega']:.4f}" if not np.isnan(greek_values['Vega']) else "N/A")
    st.metric(label="Rho Put", value=f"{greek_values['Rho Put']:.4f}" if not np.isnan(greek_values['Rho Put']) else "N/A")

with col3:
    st.metric(label="Theta Call (per year)", value=f"{greek_values['Theta Call (per year)']:.4f}" if not np.isnan(greek_values['Theta Call (per year)']) else "N/A")
    st.metric(label="Theta Put (per year)", value=f"{greek_values['Theta Put (per year)']:.4f}" if not np.isnan(greek_values['Theta Put (per year)']) else "N/A")
    # Optional: Display Theta per day
    if not np.isnan(greek_values['Theta Call (per year)']):
        st.caption(f"Call Theta (per day): {greek_values['Theta Call (per year)']/365:.4f}")
    if not np.isnan(greek_values['Theta Put (per year)']):
        st.caption(f"Put Theta (per day): {greek_values['Theta Put (per year)']/365:.4f}")


st.write("---")

# --- 3D Plotting Section (within an expander for cleaner layout) ---
with st.expander("View 3D Option Price Surfaces", expanded=False):
    st.write("## 3D Option Price Surfaces")
    st.write("These plots show how option prices change with varying Asset Price (S0) and Volatility (σ), while Strike Price (K), Interest Rate (r), and Time to Expiration (T) are held constant at the values from the sidebar.")

    # Fixed Parameters for the Graphs (using values from sidebar for K, r, T)
    # Using the K, r, T from the sidebar inputs for consistency in plots
    # K, r, T are already defined from sidebar inputs

    # Define Ranges for S0 and Volatility (for X and Y axes)
    # These ranges can be adjusted via Streamlit inputs if desired for more interactivity
    s0_plot_range = np.linspace(S0 * 0.8, S0 * 1.2, 50)  # Asset Price (S0) around current input
    volatility_plot_range = np.linspace(0.05, 0.5, 50) # Volatility (sigma) from 5% to 50%

    # Create a meshgrid for S0 and Volatility
    S0_grid, Vol_grid = np.meshgrid(s0_plot_range, volatility_plot_range)

    # Initialize Z-axis grids for Call and Put prices for the plots
    Call_Price_grid = np.zeros_like(S0_grid)
    Put_Price_grid = np.zeros_like(S0_grid)

    # Iterate through the grids to calculate prices for the plots
    for i in range(S0_grid.shape[0]):
        for j in range(S0_grid.shape[1]):
            current_S0 = S0_grid[i, j]
            current_Vol = Vol_grid[i, j]

            # Calculate Call and Put prices for the current (S0, Volatility) pair
            Call_Price_grid[i, j] = call_price_bsm(current_S0, K, r, current_Vol, T)
            Put_Price_grid[i, j] = put_price_bsm(current_S0, K, r, current_Vol, T)

    # Plot 1: Call Option Price Surface
    fig_call = plt.figure(figsize=(10, 8))
    ax_call = fig_call.add_subplot(111, projection='3d')
    surface_call = ax_call.plot_surface(S0_grid, Vol_grid, Call_Price_grid, cmap='viridis')
    ax_call.set_xlabel('Asset Price (S0)')
    ax_call.set_ylabel('Volatility (σ)')
    ax_call.set_zlabel('Call Option Price')
    ax_call.set_title('European Call Option Price Surface')
    fig_call.colorbar(surface_call, shrink=0.5, aspect=5)
    st.pyplot(fig_call) # Display the matplotlib figure in Streamlit

    # Plot 2: Put Option Price Surface
    fig_put = plt.figure(figsize=(10, 8))
    ax_put = fig_put.add_subplot(111, projection='3d')
    surface_put = ax_put.plot_surface(S0_grid, Vol_grid, Put_Price_grid, cmap='plasma')
    ax_put.set_xlabel('Asset Price (S0)')
    ax_put.set_ylabel('Volatility (σ)')
    ax_put.set_zlabel('Put Option Price')
    ax_put.set_title('European Put Option Price Surface')
    fig_put.colorbar(surface_put, shrink=0.5, aspect=5)
    st.pyplot(fig_put) # Display the matplotlib figure in Streamlit

    st.write("---")




def find_implied_volatility(option_type, S0, K, r, T, market_price,
                            low_vol=0.001, high_vol=5.0, tolerance=1e-5, max_iterations=1000):
    """
    Finds the implied volatility using the bisection method.

    Args:
        option_type (str): 'call' or 'put'.
        S0 (float): Current asset price.
        K (float): Strike price.
        r (float): Risk-free interest rate.
        T (float): Time until expiration (in years).
        market_price (float): Observed market price of the option.
        low_vol (float): Lower bound for volatility search.
        high_vol (float): Upper bound for volatility search.
        tolerance (float): Desired accuracy for the implied volatility.
        max_iterations (int): Maximum number of iterations for the bisection method.

    Returns:
        float: Implied volatility, or np.nan if not found.
    """
    if T <= 0:
        return np.nan

    # Function to calculate the difference between BSM price and market price
    def price_difference(s_val):
        if option_type == 'call':
            return call_price_bsm(S0, K, r, s_val, T) - market_price
        else: # 'put'
            return put_price_bsm(S0, K, r, s_val, T) - market_price

    # Check initial bounds
    price_low = price_difference(low_vol)
    price_high = price_difference(high_vol)

    if price_low * price_high > 0:
        # If signs are the same, the root is not bracketed.
        # This can happen if the market price is too high or too low for any volatility.
        # Try to expand the search range or return NaN.
        # For simplicity, we'll just return NaN here.
        return np.nan

    for _ in range(max_iterations):
        mid_vol = (low_vol + high_vol) / 2
        price_mid = price_difference(mid_vol)

        if abs(price_mid) < tolerance:
            return mid_vol

        if price_low * price_mid < 0:
            high_vol = mid_vol
        else:
            low_vol = mid_vol
            price_low = price_mid # Update price_low to reflect the new low_vol

    return np.nan # Implied volatility not found within max_iterations

# --- Streamlit Application Layout for Implied Volatility Calculator ---

st.title("Implied Volatility Calculator")
st.write("Enter the observed market price of an option along with its parameters to calculate its implied volatility.")

st.write("---")

# Input fields for option parameters and market price
col_left, col_right = st.columns(2)

with col_left:
    option_type_selection = st.radio("Option Type", ('Call', 'Put'))
    market_price = st.number_input("Observed Market Price", min_value=0.01, value=5.0, step=0.01, format="%.2f")
    S0 = st.number_input("S0", min_value=0.01, value=100.0, step=0.1)
    K = st.number_input("K", min_value=0.01, value=100.0, step=0.1)

with col_right:
    r = st.number_input("r", min_value=0.0, value=0.05, step=0.001, format="%.4f")
    T = st.number_input("T", min_value=0.001, value=1.0, step=0.01)
    st.markdown("---") # Visual separator
    st.write("Advanced Bisection Settings (Optional)")
    low_vol_bound = st.number_input("Lower Volatility Bound", min_value=0.001, value=0.001, step=0.001, format="%.3f")
    high_vol_bound = st.number_input("Upper Volatility Bound", min_value=0.01, value=2.0, step=0.01, format="%.2f")


if st.button("Calculate Implied Volatility"):
    if T <= 0 or S0 <= 0 or K <= 0 or market_price <= 0:
        st.error("Please ensure all input values (Asset Price, Strike Price, Time, Market Price) are positive.")
    elif low_vol_bound >= high_vol_bound:
        st.error("Lower Volatility Bound must be less than Upper Volatility Bound.")
    else:
        with st.spinner("Calculating implied volatility..."):
            implied_vol = find_implied_volatility(
                option_type_selection.lower(), S0, K, r, T, market_price,
                low_vol=low_vol_bound, high_vol=high_vol_bound
            )

            if not np.isnan(implied_vol):
                st.success(f"**Implied Volatility (σ):** {implied_vol:.4f} ({implied_vol*100:.2f}%)")
                # Verify the price with the calculated implied vol
                if option_type_selection.lower() == 'call':
                    verified_price = call_price_bsm(S0, K, r, implied_vol, T)
                else:
                    verified_price = put_price_bsm(S0, K, r, implied_vol, T)
                st.info(f"BSM price with this volatility: ${verified_price:.2f} (Target: ${market_price:.2f})")
            else:
                st.warning("Could not find implied volatility. Please check your inputs or adjust the volatility bounds.")

st.write("---")