# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:38:53 2024

@author: htchen
"""
#14846034 曾肇瀚

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


pts = 50
x = np.linspace(-2, 2, pts)
y = np.zeros(x.shape)

# square wave
pts2 = pts // 2
y[0:pts2] = -1
y[pts2:] = 1

# sort x
argidx = np.argsort(x)
x = x[argidx]
y = y[argidx]

T0 = np.max(x) - np.min(x)
f0 = 1.0 / T0
omega0 = 2.0 * np.pi * f0

# ---------- Step 1: 建構設計矩陣 X ----------
n = 5
# φ(x) = [1, cos(ω0 x), ..., cos(n ω0 x), sin(ω0 x), ..., sin(n ω0 x)]
cols = [np.ones_like(x)]
cols += [np.cos(k * omega0 * x) for k in range(1, n + 1)]
cols += [np.sin(k * omega0 * x) for k in range(1, n + 1)]
X = np.column_stack(cols)             # 形狀: (m, 1+2n)

# ---------- Step 2: X 的短式 SVD：X = U Σ V^T ----------
U, Sigma, V = mysvd(X)                # U:(m,r), Σ:(r,r), V:(p,r) 其中 p=1+2n

# ---------- Step 3: 最小平方解 a = X^+ y = V Σ^{-1} U^T y ----------
a = V @ np.linalg.inv(Sigma) @ (U.T @ y)

y_bar = X @ a
plt.plot(x, y_bar, 'g-', label='predicted values') 
plt.plot(x, y, 'b-', label='true values')
plt.xlabel('x')
plt.xlabel('y')
plt.legend()
plt.show()


