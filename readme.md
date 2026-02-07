# Markov Switching Multifractal (MSM) Volatility Model

## Overview
This project implements a **Markov Switching Multifractal (MSM)** model to analyze and forecast volatility for Bitcoin/USD.

The model captures the multi-frequency nature of financial volatility by decomposing it into distinct components that switch between states. It is specifically calibrated to estimate **Value at Risk (VaR)** and **Expected Shortfall (ES)**, providing robust risk metrics for extreme market conditions.

## Key Features
- **Regime-Switching Filter:** A recursive filter to estimate state probabilities based on historical returns.
- **Monte Carlo Simulation:** Generates 10,000+ future return paths based on current state probabilities.
- **Risk Metrics:** Calculates 1-day VaR and ES at 90%, 95%, and 99% confidence levels.
- **Vectorized Implementation:** Optimized Python code using NumPy and Pandas for high-performance simulation.

## Project Structure

```text
.
├── Data/
│   └── BTCUSD2014-2025.csv    # Historical price data
├── volatility_model.py        # Main model logic and simulation
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation