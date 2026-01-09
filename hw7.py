# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:04:38 2021
@author: 14846034-zhaohan
"""

# If this script is not run under spyder IDE, comment the following two lines.
# (safe version)


import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def scatter_pts_2d(x, y):
    xmax = np.max(x)
    xmin = np.min(x)
    xgap = (xmax - xmin) * 0.2
    xmin -= xgap
    xmax += xgap

    ymax = np.max(y)
    ymin = np.min(y)
    ygap = (ymax - ymin) * 0.2
    ymin -= ygap
    ymax += ygap

    return xmin, xmax, ymin, ymax

# --- load data (robust path) ---
csv_path = Path(__file__).resolve().parent / 'data' / 'hw7.csv'
dataset = pd.read_csv(csv_path).to_numpy(dtype=np.float64)
x = dataset[:, 0]
y = dataset[:, 1]

# parameters for our two runs of gradient descent
w0_init = np.array([-0.1607108,  2.0808538,  0.3277537, -1.5511576], dtype=np.float64)

alpha = 0.05
max_iters = 500

# =========================================================
# (1) Analytic gradient
# J(w0,w1,w2,w3) = sum( y[i] - w0 - w1*sin(w2*x[i] + w3) )^2
# =========================================================
w = w0_init.copy()
for _ in range(1, max_iters):
    s = np.sin(w[2] * x + w[3])
    c = np.cos(w[2] * x + w[3])
    y_hat = w[0] + w[1] * s
    e = y - y_hat

    g0 = -2.0 * np.sum(e)
    g1 = -2.0 * np.sum(e * s)
    g2 = -2.0 * np.sum(e * (w[1] * c * x))
    g3 = -2.0 * np.sum(e * (w[1] * c))

    grad = np.array([g0, g1, g2, g3], dtype=np.float64)
    w = w - alpha * grad

w_analytic = w.copy()

xmin, xmax, ymin, ymax = scatter_pts_2d(x, y)
xt = np.linspace(xmin, xmax, 100)
yt1 = w_analytic[0] + w_analytic[1] * np.sin(w_analytic[2] * xt + w_analytic[3])

# =========================================================
# (2) Numeric gradient (finite difference)
# =========================================================
w = w0_init.copy()

def cost(wv):
    return np.sum((y - (wv[0] + wv[1] * np.sin(wv[2] * x + wv[3])))**2)

for _ in range(1, max_iters):
    eps = 1e-6
    grad = np.zeros_like(w, dtype=np.float64)
    J0 = cost(w)

    for k in range(4):
        w_eps = w.copy()
        w_eps[k] += eps
        grad[k] = (cost(w_eps) - J0) / eps  # forward difference

    w = w - alpha * grad

w_numeric = w.copy()
yt2 = w_numeric[0] + w_numeric[1] * np.sin(w_numeric[2] * xt + w_numeric[3])

# ---------------- plot ----------------
fig = plt.figure(dpi=288)
plt.scatter(x, y, color='k', edgecolor='w', linewidth=0.9, s=60, zorder=3)
plt.plot(xt, yt1, linewidth=4, c='b', zorder=0, label='Analytic method')
plt.plot(xt, yt2, linewidth=2, c='r', zorder=0, label='Numeric method')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.show()
