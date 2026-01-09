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

# -----------------------------
# Gaussian (RBF) kernel
# K(x,z) = exp( -||x-z||^2 / (2*sigma^2) )
# -----------------------------
def rbf_kernel(X, Z, sigma=2.0):
    # X: (n,d), Z: (m,d) -> K: (n,m)
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    Xn = np.sum(X * X, axis=1, keepdims=True)       # (n,1)
    Zn = np.sum(Z * Z, axis=1, keepdims=True).T     # (1,m)
    D2 = Xn + Zn - 2.0 * (X @ Z.T)
    return np.exp(-D2 / (2.0 * sigma * sigma))

# ---------- load data (robust path) ----------
csv_path = Path(__file__).resolve().parent / 'data' / 'hw8.csv'
hw8_dataset = pd.read_csv(csv_path).to_numpy(dtype=np.float64)

X0 = hw8_dataset[:, 0:2]   # (N,2)
y  = hw8_dataset[:, 2]     # labels

# ensure labels are +/-1
y = np.where(y > 0, 1.0, -1.0)

# =========================================================
# write your code here
# Kernel Ridge Classification (binary)
# alpha = (K + lam I)^-1 y
# predict: sign(Kt @ alpha)
# =========================================================
sigma = 2.5    # <<< 可調：越大越平滑、區塊越大；越小越貼資料
lam   = 1e-2   # <<< 可調：越大越平滑、越不容易過擬合

K = rbf_kernel(X0, X0, sigma=sigma)                # (N,N)
alpha = la.pinv(K + lam * np.eye(K.shape[0])) @ y   # (N,)

# =========================================================
# plot
# =========================================================
fig = plt.figure(dpi=288)

# =========================================================
# write your code here
# 畫出分類邊界線及著色
# =========================================================
x1_min, x1_max = X0[:, 0].min() - 0.8, X0[:, 0].max() + 0.8
x2_min, x2_max = X0[:, 1].min() - 0.8, X0[:, 1].max() + 0.8

xx1, xx2 = np.meshgrid(
    np.linspace(x1_min, x1_max, 500),
    np.linspace(x2_min, x2_max, 500)
)
grid = np.c_[xx1.ravel(), xx2.ravel()]             # (G,2)

Kt = rbf_kernel(grid, X0, sigma=sigma)             # (G,N)
score = (Kt @ alpha).reshape(xx1.shape)            # real-valued decision score
pred  = np.sign(score)
pred[pred == 0] = 1

# region shading
plt.contourf(xx1, xx2, pred, levels=[-1, 0, 1], alpha=0.25)

# decision boundary line (score = 0)
plt.contour(xx1, xx2, score, levels=[0], colors='k', linewidths=2)

# scatter
plt.plot(X0[y == 1, 0],  X0[y == 1, 1], 'r.', label='$\\omega_1$')
plt.plot(X0[y == -1, 0], X0[y == -1, 1], 'b.', label='$\\omega_2$')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('equal')
plt.xlim([x1_min, x1_max])
plt.ylim([x2_min, x2_max])
plt.legend()
plt.show()
