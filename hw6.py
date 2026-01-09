# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:04:38 2021
@author: 14846034-zhaohan
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# calculate the eigenvalues and eigenvectors of a squared matrix
# the eigenvalues are decreasing ordered
def myeig(A, symmetric=False):
    if symmetric:
        lambdas, V = np.linalg.eigh(A)
    else:
        lambdas, V = np.linalg.eig(A)
    lambdas_real = np.real(lambdas)
    sorted_idx = lambdas_real.argsort()[::-1]
    return lambdas[sorted_idx], V[:, sorted_idx]


# -------------------------
# generate 2 classes
# -------------------------
np.random.seed(0)  # 固定結果（可拿掉）

mean1 = np.array([0, 5])
sigma1 = np.array([[0.3, 0.2], [0.2, 1]])
N1 = 200
X1 = np.random.multivariate_normal(mean1, sigma1, N1)

mean2 = np.array([3, 4])
sigma2 = np.array([[0.3, 0.2], [0.2, 1]])
N2 = 100
X2 = np.random.multivariate_normal(mean2, sigma2, N2)

m1 = np.mean(X1, axis=0, keepdims=True)
m2 = np.mean(X2, axis=0, keepdims=True)

# -------------------------
# Fisher LDA (2-class)
# -------------------------
Sw = (X1 - m1).T @ (X1 - m1) + (X2 - m2).T @ (X2 - m2)
md = (m1 - m2).reshape(2, 1)
Sb = md @ md.T

A = la.pinv(Sw) @ Sb
lambdas, V = myeig(A, symmetric=False)
w = np.real(V[:, 0]).reshape(2,)
w_unit = w / la.norm(w)

m0 = ((m1 + m2) * 0.5).reshape(2,)  # midpoint

# -------------------------
# plotting
# -------------------------
plt.figure(dpi=288)

# original samples
plt.plot(X1[:, 0], X1[:, 1], 'r.', ms=3, label='class 1')
plt.plot(X2[:, 0], X2[:, 1], 'g.', ms=3, label='class 2')

# means
plt.plot(m1[0, 0], m1[0, 1], 'kx', ms=8, mew=2)
plt.plot(m2[0, 0], m2[0, 1], 'kx', ms=8, mew=2)



# projection points onto w, then shift to a parallel line (for visualization)
perp = np.array([-w_unit[1], w_unit[0]])  # unit perpendicular
shift = -3.0                               # 調這個讓投影線在「下方」更像作業圖
base = m0 + shift * perp

z1 = (X1 - m0) @ w_unit
z2 = (X2 - m0) @ w_unit

# ---- draw projection as thick segments (more like the homework figure) ----
qlo, qhi = 5, 95  # 可改 10, 90 讓線段更短更像範例

z1_lo, z1_hi = np.percentile(z1, [qlo, qhi])
z2_lo, z2_hi = np.percentile(z2, [qlo, qhi])

seg1_a = base + z1_lo * w_unit
seg1_b = base + z1_hi * w_unit
seg2_a = base + z2_lo * w_unit
seg2_b = base + z2_hi * w_unit

plt.plot([seg1_a[0], seg1_b[0]], [seg1_a[1], seg1_b[1]], 'r-', lw=5)
plt.plot([seg2_a[0], seg2_b[0]], [seg2_a[1], seg2_b[1]], 'g-', lw=5)


plt.axis('equal')
plt.grid(True)
plt.show()
