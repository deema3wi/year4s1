# Lab 10 – Task 1
# Forecasting with trend (trend-adjusted exponential smoothing / Holt method)
#
# Formulas (t >= 2):
#   F_t   = α·A_(t-1) + (1-α)·(F_(t-1) + T_(t-1))
#   T_t   = β·(F_t - F_(t-1)) + (1-β)·T_(t-1)
#   FIT_t = F_t + T_t
#
# One-step-ahead forecast after the last observation A_n:
#   Forecast for period n+1 is FIT_(n+1)

import numpy as np
import matplotlib.pyplot as plt

# Input data (edit these)
A = np.array([15.0, 18.0, 20.0, 23.0, 26.0, 28.0, 31.0, 33.0, 36.0], dtype=float)  # A_1..A_n
alpha = 0.3
beta  = 0.4
F1 = 14.0  # initial level forecast
T1 = 2.0  # initial trend

n = len(A)
F = np.full(n + 2, np.nan)
T = np.full(n + 2, np.nan)
FIT = np.full(n + 2, np.nan)

F[1] = F1
T[1] = T1
FIT[1] = F1 + T1

for t in range(2, n + 2):          # t = 2..n+1
    A_prev = A[t - 2]              # A_(t-1)
    F[t] = alpha * A_prev + (1 - alpha) * (F[t - 1] + T[t - 1])
    T[t] = beta * (F[t] - F[t - 1]) + (1 - beta) * T[t - 1]
    FIT[t] = F[t] + T[t]

print("Forecast for period", n + 1, "=", FIT[n + 1])

# Plot actual vs forecast (including forecast for n+1)
t_actual = np.arange(1, n + 1)
t_fit = np.arange(1, n + 2)

plt.figure()
plt.plot(t_actual, A, marker="o", label="Actual A_t")
plt.plot(t_fit, FIT[1:n+2], marker="o", label="Forecast incl. trend (FIT_t)")
plt.xlabel("t (period)")
plt.ylabel("Value")
plt.title("Trend-adjusted exponential smoothing (Holt)")
plt.legend()
plt.show()
