# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:04:38 2021

@author: htchen
"""
#14846034 曾肇瀚
# If this script is not run under spyder IDE, comment the following two lines.
#from IPython import get_ipython
#get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd

def scatter_pts_2d(x, y):
    # set plotting limits
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

    return xmin,xmax,ymin,ymax

dataset = pd.read_csv(r'C:\Users\ASUS\Downloads\OneDrive_1_2025-10-31\hw7.csv').to_numpy(dtype = np.float64)
x = dataset[:, 0]
y = dataset[:, 1]

# parameters for our two runs of gradient descent
w = np.array([-0.1607108,  2.0808538,  0.3277537, -1.5511576])

alpha = 0.05
max_iters = 500
def cost(w, x, y):
    y_hat = w[0] + w[1] * np.sin(w[2] * x + w[3])
    e = y - y_hat
    return np.sum(e**2)

#     J(w0, w1, w2, w3) = sum(y[i] - w0 - w1 * sin(w2 * x[i] + w3))^2
for _ in range(1, max_iters):
       # 預測與殘差
    s = np.sin(w[2] * x + w[3])      # sin(w2 x + w3)
    c = np.cos(w[2] * x + w[3])      # cos(w2 x + w3)
    y_hat = w[0] + w[1] * s
    e = y - y_hat                    # e_i

    # 解析法梯度（偏導）
    # dJ/dw0 = -2 Σ e_i
    g0 = -2.0 * np.sum(e)
    # dJ/dw1 = -2 Σ e_i * sin(...)
    g1 = -2.0 * np.sum(e * s)
    # dJ/dw2 = -2 Σ e_i * w1 * x_i * cos(...)
    g2 = -2.0 * np.sum(e * w[1] * x * c)
    # dJ/dw3 = -2 Σ e_i * w1 * cos(...)
    g3 = -2.0 * np.sum(e * w[1] * c)

    grad = np.array([g0, g1, g2, g3])

    # 更新規則
    w = w - alpha * grad



xmin,xmax,ymin,ymax = scatter_pts_2d(x, y)
xt = np.linspace(xmin, xmax, 100)
yt1 = w[0] + w[1] * np.sin(w[2] * xt + w[3])

w = np.array([-0.1607108,  2.0808538,  0.3277537, -1.5511576])
eps = 1e-8
for _ in range(1, max_iters):
    J0 = cost(w, x, y)
    grad = np.zeros_like(w)

    # 對每一個參數用數值微分
    for k in range(len(w)):
        w_eps = w.copy()
        w_eps[k] += eps
        J_eps = cost(w_eps, x, y)
        grad[k] = (J_eps - J0) / eps

    # 更新
    w = w - alpha * grad
    

xt = np.linspace(xmin, xmax, 100)
yt2 = w[0] + w[1] * np.sin(w[2] * xt + w[3])

# plot x vs y; xt vs yt1; xt vs yt2 
fig = plt.figure(dpi=288)
plt.scatter(x, y, color='k', edgecolor='w', linewidth=0.9, s=60, zorder=3)
plt.plot(xt, yt1, linewidth=4, c='b', zorder=0, label='Analytic method')
plt.plot(xt, yt2, linewidth=2, c='r', zorder=0, label='Numeric method')
plt.xlim([xmin,xmax])
plt.ylim([ymin,ymax])
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.show()
