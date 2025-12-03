# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 09:37:05 2021

@author: htchen
"""
#14846034 曾肇瀚
# If this script is not run under spyder IDE, comment the following two lines.
#from IPython import get_ipython
#get_ipython().run_line_magic('reset', '-sf')

import math
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd

# calculate the eigenvalues and eigenvectors of a squared matrix
# the eigenvalues are decreasing ordered
def myeig(A, symmetric=False):
    if symmetric:
        lambdas, V = np.linalg.eigh(A)
    else:
        lambdas, V = np.linalg.eig(A)
    # lambdas, V may contain complex value
    lambdas_real = np.real(lambdas)
    sorted_idx = lambdas_real.argsort()[::-1] 
    return lambdas[sorted_idx], V[:, sorted_idx]

# SVD: A = U * Sigma * V^T
# V: eigenvector matrix of A^T * A; U: eigenvector matrix of A * A^T 
def mysvd(A):
    lambdas, V = myeig(A.T @ A, symmetric=True)
    lambdas, V = np.real(lambdas), np.real(V)
    # if A is full rank, no lambda value is less than 1e-6 
    # append a small value to stop rank check
    lambdas = np.append(lambdas, 1e-12)
    rank = np.argwhere(lambdas < 1e-6).min()
    lambdas, V = lambdas[0:rank], V[:, 0:rank]
    U = A @ V / np.sqrt(lambdas)
    Sigma = np.diag(np.sqrt(lambdas))
    return U, Sigma, V

def row_norm_square(X):
    return np.sum(X * X, axis=1)

# gaussian weight array g=[ g_1 g_2 ... g_m ]
# g_i = exp(-0.5 * ||x_i - c||^2 / sigma^2)
def gaussian_weight(X, c, sigma=1.0):
    s = 0.5 / sigma / sigma;
    norm2 = row_norm_square(X - c)
    g = np.exp(-s * norm2)
    return g

# xt: a sample in Xt
# yt: predicted value of f(xt)
# yt = (X.T @ G(xt) @ X)^-1 @ X.T @ G(xt) @ y
def predict(X, y, Xt, sigma=1.0):
    ntest = Xt.shape[0] # number of test samples 
    yt = np.zeros(ntest)
    for xi in range(ntest):
        c = Xt[xi, :]
        g = gaussian_weight(X, c, sigma) # diagonal elements in G
        G = np.diag(g)
        w = la.pinv(X.T @ G @ X) @ X.T @ G @ y
        yt[xi] = c @ w
    return yt

# Xs: m x n matrix; 
# m: pieces of sample
# K: m x m kernel matrix
# K[i,j] = exp(-c(|xt_i|^2 + |xs_j|^2 -2(xt_i)^T @ xs_j)) where c = 0.5 / sigma^2
# 更多實作說明, 參考課程oneonte筆記

def calc_gaussian_kernel(Xt, Xs, sigma=1):
    nt, _ = Xt.shape # pieces of Xt
    ns, _ = Xs.shape # pieces of Xs
    
    norm_square = row_norm_square(Xt)
    F = np.tile(norm_square, (ns, 1)).T
    
    norm_square = row_norm_square(Xs)
    G = np.tile(norm_square, (nt, 1))
    
    E = F + G - 2.0 * Xt @ Xs.T
    s = 0.5 / (sigma * sigma)
    K = np.exp(-s * E)
    return K

# n: degree of polynomial
# generate X=[1 x x^2 x^3 ... x^n]
# m: pieces(rows) of data(X)
# X is a m x (n+1) matrix
def poly_data_matrix(x: np.ndarray, n: int):
    m = x.shape[0]
    X = np.zeros((m, n + 1))
    X[:, 0] = 1.0
    for deg in range(1, n + 1):
        X[:, deg] = X[:, deg - 1] * x
    return X

hw5_csv = pd.read_csv(r'C:\Users\ASUS\Downloads\OneDrive_1_2025-10-31\hw5.csv')
hw5_dataset = hw5_csv.to_numpy(dtype = np.float64)

hours = hw5_dataset[:, 0]
sulfate = hw5_dataset[:, 1]





# ---------------------------------------------------------
# (1) 濃度 vs 時間：散佈圖 + 多項式迴歸曲線
# ---------------------------------------------------------
plt.figure()

# 原始資料
plt.plot(hours, sulfate, 'ko', label='data')

# 這裡選擇一個「多項式回歸」作為我們的迴歸方法，例：三次多項式
deg = 3
X = poly_data_matrix(hours, deg)
U, Sigma, V = mysvd(X)
# least square 解：a = V Σ^{-1} U^T y
a = V @ np.linalg.inv(Sigma) @ (U.T @ sulfate)

# 為了畫平滑曲線，我們在時間軸上取更多點
t_grid = np.linspace(hours.min(), hours.max(), 200)
X_grid = poly_data_matrix(t_grid, deg)
sulfate_pred = X_grid @ a

plt.plot(t_grid, sulfate_pred, 'b-', label=f'poly deg={deg} regression')

plt.title('Sulfate concentration vs time')
plt.xlabel('time in hours')
plt.ylabel('sulfate concentration (times $10^{-4}$)')
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# (2) log(濃度) vs log(時間)：散佈圖 + 線性迴歸直線
# ---------------------------------------------------------

# 只取正值（時間與濃度都應該是正的，這裡保險起見）
mask = (hours > 0) & (sulfate > 0)
hours_pos = hours[mask]
sulfate_pos = sulfate[mask]

log_t = np.log(hours_pos)
log_c = np.log(sulfate_pos)

plt.figure()

# log-log 的散佈圖
plt.plot(log_t, log_c, 'ko', label='log-log data')

# 在 log-log 空間做線性回歸：log_c = b0 + b1 * log_t
X_log = poly_data_matrix(log_t, 1)  # [1, log_t]
U_log, Sigma_log, V_log = mysvd(X_log)
a_log = V_log @ np.linalg.inv(Sigma_log) @ (U_log.T @ log_c)

# 畫線
xg = np.linspace(log_t.min(), log_t.max(), 200)
Xg_log = poly_data_matrix(xg, 1)
yg = Xg_log @ a_log

plt.plot(xg, yg, 'b-', label='linear regression in log-log')

plt.title('log(sulfate concentration) vs log(time)')
plt.xlabel('log(time in hours)')
plt.ylabel('log(sulfate concentration  (times $10^{-4}$))')
plt.legend()
plt.tight_layout()
plt.show()
