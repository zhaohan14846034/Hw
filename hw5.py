# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 09:37:05 2021

@author: 14846034-zhaohan
"""
# If this script is not run under spyder IDE, comment the following two lines.


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

from pathlib import Path
base_dir = Path(__file__).resolve().parent
hw5_csv = pd.read_csv(base_dir / 'data' / 'hw5.csv')

hw5_dataset = hw5_csv.to_numpy(dtype = np.float64)

hours = hw5_dataset[:, 0]
sulfate = hw5_dataset[:, 1]

# --------------------------
# 基本資料圖
plt.figure()
plt.plot(hours, sulfate, 'ko', label='data')
plt.title('concentration vs time')
plt.xlabel('time in hours')
plt.ylabel('sulfate concentration (times $10^{-4}$)')
plt.grid(True)
plt.legend()
plt.show()

# --------------------------
# log-log 圖（避免 log(0)）
eps = 1e-12
x_log = np.log(hours + eps)
y_log = np.log(sulfate + eps)

plt.figure()
plt.plot(hours, sulfate, 'ko', label='data')
plt.xscale("log")
plt.yscale("log")
plt.title('concentration vs time (log-log scale)')
plt.xlabel('time in hours (log)')
plt.ylabel('sulfate concentration (log)')
plt.grid(True, which='both')
plt.legend()
plt.show()

# --------------------------
# (1) Power-law 擬合： y = alpha * x^beta
# log(y) = log(alpha) + beta * log(x)
Xlog = poly_data_matrix(x_log, 1)              # [1, log(x)]
w = la.pinv(Xlog) @ y_log                      # w[0]=log(alpha), w[1]=beta
alpha = np.exp(w[0])
beta = w[1]
print(f'[power-law fit] alpha={alpha:.6g}, beta={beta:.6g}')

# 產生平滑曲線
xg = np.linspace(hours.min(), hours.max(), 400)
yg = alpha * (xg ** beta)

plt.figure()
plt.plot(hours, sulfate, 'ko', label='data')
plt.plot(xg, yg, 'r-', label=f'fit: y={alpha:.3g} x^{beta:.3g}')
plt.title('power-law fit (linear scale)')
plt.xlabel('time in hours')
plt.ylabel('sulfate concentration (times $10^{-4}$)')
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(hours, sulfate, 'ko', label='data')
plt.plot(xg, yg, 'r-', label='power-law fit')
plt.xscale("log")
plt.yscale("log")
plt.title('power-law fit (log-log scale)')
plt.xlabel('time in hours (log)')
plt.ylabel('sulfate concentration (log)')
plt.grid(True, which='both')
plt.legend()
plt.show()

# --------------------------
# (2) 可選：局部加權線性回歸（LWLR）
# 注意：predict() 裡距離是用 X(含 bias) 算的，這裡用 [1, x] 跑即可
X = np.column_stack([np.ones_like(hours), hours])
y = sulfate.copy()

Xt = np.column_stack([np.ones_like(xg), xg])

sigma = 50.0   # 可調：越小越「貼點」但越抖；越大越平滑
y_lw = predict(X, y, Xt, sigma=sigma)

plt.figure()
plt.plot(hours, sulfate, 'ko', label='data')
plt.plot(xg, y_lw, 'g-', label=f'LWLR (sigma={sigma})')
plt.title('Locally Weighted Linear Regression')
plt.xlabel('time in hours')
plt.ylabel('sulfate concentration (times $10^{-4}$)')
plt.grid(True)
plt.legend()
plt.show()





# plt.title('concentration vs time')
# plt.xlabel('time in hours')
# plt.ylabel('sulfate concentration (times $10^{-4}$)')



# plt.xscale("log")
# plt.yscale("log")
# plt.title('concentration vs time (log-log scale)')
# plt.xlabel('time in hours')
# plt.ylabel('sulfate concentration  (times $10^{-4}$)')


