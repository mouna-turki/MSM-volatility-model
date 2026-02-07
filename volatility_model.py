# HW 43 

import math, statistics, pandas as pd, matplotlib.pyplot as plt, numpy as np
import random
import scipy.stats, statsmodels.api as sm
from scipy.stats import norm

# ==============================================================================
# SECTION 1: USER INPUTS (EDIT THIS SECTION ONLY)
# ==============================================================================

# 1. Data Settings
filename = 'Data/BTCUSD2014-2025.csv'       # Name of your CSV file
cutoff_date = '12/31/2020'             # The last in-sample date

# 2. Estimated Parameters (From previous exercises/questions)
a0 = 0.001568
a1 = -0.08789
a2 = 0.2182
m0 = 1.801
gammabar = 0.778
b = 12.61
sigma = 0.03757

# 3. Simulation Settings
N = 10000        # Number of simulations (usually 10,000)
seed_val = 1     # Random seed (keeps results consistent)
kbar = 2         # Number of volatility components (Usually 2 for these exams)

# ==============================================================================
# SECTION 2: CALCULATIONS & LOGIC (DO NOT EDIT BELOW THIS LINE)
# ==============================================================================

# --- A. Data Preparation ---
data = pd.read_csv(filename)
r = data.Return
data.Date = pd.to_datetime(data.Date, format='%m/%d/%Y')
Tp = len(r)
x = np.arange(0, Tp)
index = x[data.Date == cutoff_date]
T = int(index[0])
print(f'T (Last In-Sample Index) = {T}')

T1 = T + 1 + 1  # Forecasting horizon
rD = [0] * T1
rM = [0] * T1

# Calculate Lagged Means
for t in range(22, T1 + 1):
    rD[t - 1] = r[t - 1]
    rM[t - 1] = statistics.mean(r[(t - 22):t])

# --- B. The Filter Function ---
def filter(theta):
    # Unpack parameters
    a0 = theta[0]; a1 = theta[1]; a2 = theta[2]
    m0 = theta[3]; gammabar = theta[4]; b = theta[5]; sigma = theta[6]
    sigma2 = sigma**2
    
    # Gamma setup
    gammas = [0] * kbar
    for k in range(0, kbar):
        gammas[k] = gammabar / b ** (kbar - k - 1)
    
    d = 2**kbar
    m = np.empty((d, kbar))
    # Matrix setup for kbar=2
    m[0, :] = [m0, m0]
    m[1, :] = [m0, 2 - m0]
    m[2, :] = [2 - m0, m0]
    m[3, :] = [2 - m0, 2 - m0]
    
    A = np.empty((d, d))
    for i in range(0, d):
        for j in range(0, d):
            A[i, j] = 1
            for k in range(0, kbar):
                if m[i, k] != m[j, k]:
                    A[i, j] = A[i, j] * gammas[k] / 2
                else:
                    A[i, j] = A[i, j] * (1 - gammas[k] / 2)
    
    mut = [a0] * T1
    lambdat = [1 / d] * d
    
    # Filtering Loop
    for t in range(0, T1):
        frt = [0] * d
        if t > 21:
            mut[t] = a0 + a1 * r[t - 1] + a2 * rM[t - 1]
        
        lambdatA = [0] * d
        # Prediction Step
        for j in range(0, d):
            for k in range(0, d):
                lambdatA[j] = lambdatA[j] + lambdat[k] * A[k, j]
        
        # Update Step
        sumd = 0
        for j in range(0, d):
            sigma2s = sigma2 * np.prod(m[j, :])
            frt[j] = norm.pdf(r[t], mut[t], sigma2s**0.5)
            sumd = sumd + frt[j] * lambdatA[j]
            
        lambdattilde = [0] * d
        for j in range(0, d):
            lambdattilde[j] = frt[j] * lambdatA[j]
        
        lambdat = lambdattilde / np.sum(lambdattilde)
        
    return lambdatA

# --- C. Run Filter and Simulation ---
thetaML = [a0, a1, a2, m0, gammabar, b, sigma]
lambdatA = filter(thetaML)
print('State Probabilities (lambdatA):', lambdatA)

np.random.seed(seed_val)
eps = np.random.normal(0, 1, N)

# Setup M states again for simulation
d = 2 ** kbar
m = np.empty((d, kbar))
m[0, :] = [m0, m0]
m[1, :] = [m0, 2 - m0]
m[2, :] = [2 - m0, m0]
m[3, :] = [2 - m0, 2 - m0]

# Choose states based on filter output
indexes = np.random.choice(d, N, replace=True, p=lambdatA)
M = m[indexes, :]

# Simulate Returns
rsim = [0] * N
sigma2 = sigma**2
for n in range(0, N):
    sigma2s = sigma2 * np.prod(M[n, :])
    rsim[n] = a0 + a1 * r[T] + a2 * statistics.mean(r[(T - 21):(T + 1)]) + sigma2s**0.5 * eps[n]

# --- D. VaR and ES Calculations ---
def VaR(alphastar):
    q = np.quantile(rsim, alphastar)
    return -q

def ES(alphastar):
    q = np.quantile(rsim, alphastar)
    S = 0; cont = 0
    for n in range(1, N):
        if rsim[n] < q:
            S = S + rsim[n]
            cont = cont + 1
    return -S / cont

# Output
VaRs = [VaR(0.10), VaR(0.05), VaR(0.01)]
ESs = [ES(0.10), ES(0.05), ES(0.01)]
dat = {'VaR': VaRs, 'ES': ESs}
df = pd.DataFrame(dat, index=['90%', '95%', '99%'])
print('\nResults:')
print(df)

# write a function that takes an integer a and returns 2*a
def double(a):
    return 2 * a