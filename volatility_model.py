"""
Markov Switching Multifractal (MSM) Volatility Model
for Bitcoin/USD Risk Analysis (VaR & Expected Shortfall).

Refactored for readability and GitHub standards.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import itertools
from dataclasses import dataclass
from typing import Tuple, List

# ==============================================================================
# CONFIGURATION & PARAMETERS
# ==============================================================================

@dataclass
class ModelParams:
    """Holds estimated model parameters."""
    a0: float = 0.001568    # Intercept
    a1: float = -0.08789    # AR(1) Coefficient
    a2: float = 0.2182      # Rolling Mean Coefficient
    m0: float = 1.801       # Multiplier
    gamma_bar: float = 0.778
    b: float = 12.61        # Frequency parameter
    sigma: float = 0.03757  # Volatility constant

# Simulation Settings
CONFIG = {
    'filename': 'Data/BTCUSD2014-2025.csv',
    'cutoff_date': '2020-12-31',
    'simulations': 10000,
    'seed': 1,
    'vol_components': 2,  # kbar
    'rolling_window': 22
}

# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def load_and_prep_data(filepath: str, cutoff_date: str, window: int) -> Tuple[pd.Series, pd.Series, int]:
    """
    Loads CSV data, formats dates, and calculates rolling features.
    
    Returns:
        returns (pd.Series): The Return column.
        rolling_means (pd.Series): 22-day rolling mean of returns.
        cutoff_index (int): The integer index of the cutoff date.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file at {filepath}")

    # clean date and set index logic
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    
    # Find cutoff index
    cutoff_indices = df.index[df['Date'] == pd.to_datetime(cutoff_date)].tolist()
    if not cutoff_indices:
        raise ValueError(f"Cutoff date {cutoff_date} not found in data.")
    cutoff_index = cutoff_indices[0]

    returns = df['Return']
    
    # Calculate rolling mean efficiently using Pandas
    # Note: The original code used a lag. We calculate rolling then shift in the loop or here.
    # Original logic: rM[t-1] = mean(r[t-22 : t]) -> This is a closed window rolling mean.
    rolling_means = returns.rolling(window=window).mean()

    return returns, rolling_means, cutoff_index


def build_transition_matrix(kbar: int, params: ModelParams) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Constructs the Transition Matrix (A) and State Multiplier Matrix (M) 
    based on the number of volatility components.
    """
    # 1. Generate State Combinations (Generalized for any kbar)
    # For kbar=2, states are combinations of m0 and (2-m0)
    states = [params.m0, 2 - params.m0]
    # Cartesian product to get all state vectors (e.g., [m0, m0], [m0, 2-m0]...)
    m_states = np.array(list(itertools.product(states, repeat=kbar)))
    
    d = len(m_states)  # 2^kbar

    # 2. Calculate Gammas
    gammas = np.zeros(kbar)
    for k in range(kbar):
        gammas[k] = params.gamma_bar / (params.b ** (kbar - k - 1))

    # 3. Build Transition Matrix A
    A = np.ones((d, d))
    for i in range(d):
        for j in range(d):
            for k in range(kbar):
                # Check if the k-th component changed between state i and state j
                if m_states[i, k] != m_states[j, k]:
                    A[i, j] *= gammas[k] / 2
                else:
                    A[i, j] *= (1 - gammas[k] / 2)
    
    return A, m_states, d


def run_msm_filter(returns: pd.Series, 
                   rolling_means: pd.Series, 
                   T_cutoff: int, 
                   params: ModelParams, 
                   kbar: int) -> np.ndarray:
    """
    Runs the recursive MSM filter to determine state probabilities at T_cutoff.
    """
    A, m_matrix, d = build_transition_matrix(kbar, params)
    
    # Initialize probabilities (uniform distribution initially)
    lambda_t = np.ones(d) / d
    
    # Pre-calculate variance multipliers for each state
    # sigma2_states[j] = sigma^2 * prod(m_state_j)
    sigma2 = params.sigma ** 2
    state_variances = sigma2 * np.prod(m_matrix, axis=1)
    state_stds = np.sqrt(state_variances)

    # Forecasting Horizon (T + 1 step ahead as per original code logic)
    horizon = T_cutoff + 2 
    
    # Filtering Loop
    # Note: Original code starts calculating Conditional Mean (mu) after t > 21
    current_lambda_pred = np.zeros(d)
    
    for t in range(horizon):
        # 1. Calculate Conditional Mean (mu_t)
        mu_t = params.a0
        if t > 21:
            # Note: accessing t-1 for lag
            r_lag = returns.iloc[t-1]
            rm_lag = rolling_means.iloc[t-1]
            mu_t += params.a1 * r_lag + params.a2 * rm_lag

        # 2. Prediction Step (A * lambda_t-1)
        # Using dot product for speed: lambda_pred = lambda_prev @ A
        # Original code: lambdatA[j] = sum(lambdat[k] * A[k, j])
        lambda_pred = np.dot(lambda_t, A)

        # 3. Update Step (Bayes Rule with Likelihood)
        # Calculate likelihood of observation r[t] given each state
        # Standard normal PDF: (x - mu) / sigma
        likelihoods = norm.pdf(returns.iloc[t], loc=mu_t, scale=state_stds)
        
        # Element-wise multiplication: P(State) * P(Data|State)
        lambda_tilde = likelihoods * lambda_pred
        
        # Normalize to sum to 1
        sum_tilde = np.sum(lambda_tilde)
        if sum_tilde == 0:
            # Fallback to prevent division by zero if densities vanish
            lambda_t = lambda_pred 
        else:
            lambda_t = lambda_tilde / sum_tilde
            
        # Store the prediction for the final return
        current_lambda_pred = lambda_pred

    return current_lambda_pred


def simulate_forecast(state_probs: np.ndarray, 
                      last_return: float, 
                      last_rolling: float, 
                      params: ModelParams, 
                      kbar: int, 
                      n_sims: int, 
                      seed: int) -> np.ndarray:
    """
    Simulates N future returns based on the filtered state probabilities.
    """
    np.random.seed(seed)
    
    # Re-generate states (cheap operation, keeps function pure)
    _, m_matrix, d = build_transition_matrix(kbar, params)
    
    # 1. Select State for each simulation based on filter probabilities
    state_indices = np.random.choice(d, size=n_sims, replace=True, p=state_probs)
    selected_m = m_matrix[state_indices] # Shape (N, kbar)
    
    # 2. Calculate Volatility for each simulation
    # sigma_sim = sqrt(sigma^2 * prod(M))
    sigma2 = params.sigma ** 2
    sim_variances = sigma2 * np.prod(selected_m, axis=1)
    sim_stds = np.sqrt(sim_variances)
    
    # 3. Generate Random Shocks
    eps = np.random.normal(0, 1, n_sims)
    
    # 4. Compute Simulated Returns
    # Forecast Mean = a0 + a1 * r[T] + a2 * rM[T]
    forecast_mean = params.a0 + (params.a1 * last_return) + (params.a2 * last_rolling)
    
    simulated_returns = forecast_mean + (sim_stds * eps)
    
    return simulated_returns


def calculate_risk_metrics(simulated_returns: np.ndarray) -> pd.DataFrame:
    """
    Calculates Value at Risk (VaR) and Expected Shortfall (ES).
    """
    alphas = [0.10, 0.05, 0.01]
    results = {}
    
    for alpha in alphas:
        # VaR is the quantile
        q = np.quantile(simulated_returns, alpha)
        var_val = -q
        
        # ES is the average of returns below the quantile
        tail_losses = simulated_returns[simulated_returns < q]
        es_val = -np.mean(tail_losses) if len(tail_losses) > 0 else 0.0
        
        results[f'{int((1-alpha)*100)}%'] = {'VaR': var_val, 'ES': es_val}
        
    return pd.DataFrame(results).T

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("--- Starting MSM Risk Analysis ---")
    
    # 1. Initialize
    params = ModelParams()
    
    # 2. Data Preparation
    print(f"Loading data from {CONFIG['filename']}...")
    try:
        r_series, rm_series, T_idx = load_and_prep_data(
            CONFIG['filename'], 
            CONFIG['cutoff_date'], 
            CONFIG['rolling_window']
        )
        print(f"Data Loaded. Last In-Sample Index (T): {T_idx}")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    # 3. Run Filter
    print("Running Regime Switching Filter...")
    final_state_probs = run_msm_filter(
        r_series, 
        rm_series, 
        T_idx, 
        params, 
        CONFIG['vol_components']
    )
    print(f"State Probabilities at T+1: {np.round(final_state_probs, 4)}")

    # 4. Simulation
    print(f"Simulating {CONFIG['simulations']} outcomes...")
    
    # Inputs for simulation are taken from the cutoff date T
    # Note: Using slice T-21:T+1 for rolling mean to match original logic approx
    # But since we have rm_series computed, we can just grab rm_series[T]
    val_r_T = r_series.iloc[T_idx]
    val_rm_T = rm_series.iloc[T_idx]

    sim_returns = simulate_forecast(
        final_state_probs,
        val_r_T,
        val_rm_T,
        params,
        CONFIG['vol_components'],
        CONFIG['simulations'],
        CONFIG['seed']
    )

    # 5. Risk Metrics
    print("\n--- Risk Metrics Results ---")
    risk_df = calculate_risk_metrics(sim_returns)
    print(risk_df)
    
    # Optional: Plotting (Commented out for GitHub suitability unless requested)
    # plt.hist(sim_returns, bins=50, alpha=0.7)
    # plt.title("Simulated Return Distribution")
    # plt.show()