# -*- coding: utf-8 -*-
"""
@author: 14846034-zhaohan

"""
import numpy as np
import numpy.linalg as la

def gram_schmidt(S1: np.ndarray):
    m, n = S1.shape
    S2 = np.zeros((m, n), dtype=np.float64)

    eps = 1e-12  # 避免除到 0

    for r in range(n):
        v = S1[:, r].astype(np.float64)

        # u_r = v_r - sum_{i=1}^{r-1} (v_r^T e_i) e_i
        u = v.copy()
        for i in range(r):
            ei = S2[:, i]
            u -= (v @ ei) * ei

        norm_u = la.norm(u)
        if norm_u < eps:
            raise ValueError(f"Column {r} is (numerically) linearly dependent; norm={norm_u}")

        # e_r = u_r / ||u_r||
        S2[:, r] = u / norm_u

    return S2


S1 = np.array([[ 7,  4,  7, -3, -9],
               [-1, -4, -4,  1, -4],
               [ 8,  0,  5, -6,  0],
               [-4,  1,  1, -1,  4],
               [ 2,  3, -5,  1,  8]], dtype=np.float64)
S2 = gram_schmidt(S1)

np.set_printoptions(precision=2, suppress=True)
print(f'S1 => \n{S1}')
print(f'S2.T @ S2 => \n{S2.T @ S2}')

"""
Expected output:
------------------
S1 => 
[[ 7.  4.  7. -3. -9.]
 [-1. -4. -4.  1. -4.]
 [ 8.  0.  5. -6.  0.]
 [-4.  1.  1. -1.  4.]
 [ 2.  3. -5.  1.  8.]]
S2.T @ S2 => 
[[ 1. -0. -0.  0.  0.]
 [-0.  1. -0. -0. -0.]
 [-0. -0.  1.  0.  0.]
 [ 0. -0.  0.  1.  0.]
 [ 0. -0.  0.  0.  1.]]
"""  