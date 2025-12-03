# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:04:38 2021

@author: htchen
"""
#14846034 曾肇瀚
#from IPython import get_ipython
#get_ipython().run_line_magic('reset', '-sf')

import math
import numpy as np
import numpy.linalg as la
import cv2
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


# class 1
mean1 = np.array([0, 5])
sigma1 = np.array([[0.3, 0.2],
                   [0.2, 1.0]])
N1 = 200
X1 = np.random.multivariate_normal(mean1, sigma1, N1)

# class 2
mean2 = np.array([3, 4])
sigma2 = np.array([[0.3, 0.2],
                   [0.2, 1.0]])
N2 = 100
X2 = np.random.multivariate_normal(mean2, sigma2, N2)

# ------------------ LDA：求 w ------------------
m1 = X1.mean(axis=0)   # 類別 1 的平均 (2,)
m2 = X2.mean(axis=0)   # 類別 2 的平均 (2,)

S1 = (X1 - m1).T @ (X1 - m1)   # 類內散佈矩陣
S2 = (X2 - m2).T @ (X2 - m2)
Sw = S1 + S2

# 兩類 LDA 的閉式解： w ∝ Sw^{-1} (m1 - m2)
w = la.inv(Sw) @ (m1 - m2)
w = w / la.norm(w)              # 正規化，方便解讀
if (m1 @ w) > (m2 @ w):
    w = -w
# ------------------ 將點投影到 w 上（一維） ------------------
# 投影係數 y = x · w
y1 = X1 @ w        # shape (N1,)
y2 = X2 @ w        # shape (N2,)

if (m1 @ w) > (m2 @ w):
    w = -w
# 選一個基準點，讓投影線畫在資料下面一點
# 先抓兩類一起的平均點
m_all = (N1 * m1 + N2 * m2) / (N1 + N2)   # shape (2,)
# 再沿著 w 的法向量往下移一段，避免壓在點雲中間
n_perp = np.array([-w[1], w[0]])         # 和 w 垂直的向量
p0 = m_all - 3.0 * n_perp                # 基準點（線上其中一點）

# 把一維投影放回 2D：p = p0 + t * w
X1_proj = p0 + np.outer(y1, w)           # (N1,2)
X2_proj = p0 + np.outer(y2, w)           # (N2,2)

# 用同一條方向畫出投影「基準線」
t_line_min = min(y1.min(), y2.min()) - 1
t_line_max = max(y1.max(), y2.max()) + 1
t_line = np.linspace(t_line_min, t_line_max, 2)
line_pts = p0 + np.outer(t_line, w)
# ------------------ 畫圖 ------------------
plt.figure(dpi=144)

# 上半部：原始 2D 資料
plt.plot(X1[:, 0], X1[:, 1], 'r.', label='class 1')
plt.plot(X2[:, 0], X2[:, 1], 'g.', label='class 2')

# 下半部：在 w 上的一維投影（排在 x 軸附近）
plt.plot(X1_proj[:, 0], X1_proj[:, 1], 'r.', alpha=0.9)
plt.plot(X2_proj[:, 0], X2_proj[:, 1], 'g.', alpha=0.9)

plt.axhline(0, color='k', linewidth=0.5)  # x 軸線，讓下面那條更明顯
plt.xlim(-6, 7)
plt.ylim(-1, 7)
plt.xlabel('x1')
plt.ylabel('x2 / projection')
plt.legend(loc='upper right')
plt.title('LDA: data and projection onto w')
plt.tight_layout()
plt.show()