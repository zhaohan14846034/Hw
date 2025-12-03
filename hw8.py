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
from sklearn.svm import SVC


hw8_csv = pd.read_csv(r'C:\Users\ASUS\Downloads\OneDrive_1_2025-10-31\hw8.csv')
hw8_dataset = hw8_csv.to_numpy(dtype = np.float64)

X0 = hw8_dataset[:, 0:2]
y = hw8_dataset[:, 2]

# -------------------------------------------------
# 使用 RBF kernel SVM 訓練分類器
# -------------------------------------------------
clf = SVC(kernel='rbf', C=5.0, gamma='scale')
clf.fit(X0, y)

# -------------------------------------------------
# 產生網格，做每個點的預測，用來畫決策區域
# -------------------------------------------------
x_min, x_max = X0[:, 0].min() - 1.0, X0[:, 0].max() + 1.0
y_min, y_max = X0[:, 1].min() - 1.0, X0[:, 1].max() + 1.0

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = clf.predict(grid)
Z = Z.reshape(xx.shape)    # 變回與網格相同的形狀

# -------------------------------------------------
# 作圖
# -------------------------------------------------
fig = plt.figure(dpi=288)

# 背景顏色（分類區域上色）
# 把標籤 -1, +1 映射成 0,1 方便著色
Z_bg = (Z > 0).astype(int)
# 0 → 淺綠, 1 → 深綠（你也可以改成自己喜歡的顏色）
from matplotlib.colors import ListedColormap
cmap_bg = ListedColormap(['#d5f5d5', '#1b5e20'])
plt.contourf(xx, yy, Z_bg, alpha=0.6, cmap=cmap_bg)

# 決策邊界線（類別切換的地方）
plt.contour(xx, yy, Z, levels=[0], colors='k', linewidths=1.0)


plt.plot(X0[y == 1, 0], X0[y == 1, 1], 'r.', label='$\omega_1$')
plt.plot(X0[y == -1, 0], X0[y == -1, 1], 'b.', label='$\omega_2$')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('equal')
plt.legend()
plt.show()

