# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 10:46:50 2021

@author: 14846034-zhaohan

"""


import math
import numpy as np
import numpy.linalg as la
import cv2
import matplotlib.pyplot as plt
import pandas as pd
figdpi = 400

from pathlib import Path

csv_path = Path(__file__).resolve().parent / 'data' / 'hw9.csv'
hw9_csv = pd.read_csv(csv_path).to_numpy(dtype=np.float64)

t = hw9_csv[:, 0] # 時間
flow_velocity = hw9_csv[:, 1] # 氣體流速
plt.figure(dpi=figdpi)
plt.plot(t, flow_velocity, 'r')
plt.title('Gas Flow Velocity')
plt.xlabel('time in seconds')
plt.ylabel('ml/sec')
plt.show()

# Integrating the gas flow velocity yields the net flow
net_vol = np.cumsum(flow_velocity) * 0.01
plt.figure(dpi=figdpi)
plt.plot(t, net_vol, 'r')
plt.title('Gas Net Flow')
plt.xlabel('time in seconds')
plt.ylabel('ml')
plt.show()

A = np.zeros((len(t), 3))
A[:, 0] = 1
A[:, 1] = t
A[:, 2] = t * t
y = net_vol
a = la.inv(A.T @ A) @ A.T @ y
trend_curve = a[0] + a[1] * t + a[2] * t * t

# ---- find data trend line & detrend ----
# 1) show net_vol with fitted trend curve
plt.figure(dpi=figdpi)
plt.plot(t, net_vol, 'r', label='net_vol (integrated)')
plt.plot(t, trend_curve, 'k--', linewidth=2, label='trend curve (LS quad)')
plt.title('Gas Net Flow + Trend')
plt.xlabel('time in seconds')
plt.ylabel('ml')
plt.legend()
plt.show()

# 2) remove trend and plot
net_vol_detrended = net_vol - trend_curve

plt.figure(dpi=figdpi)
plt.plot(t, net_vol_detrended, 'r')
plt.title('Gas Net Flow (Detrended)')
plt.xlabel('time in seconds')
plt.ylabel('ml')
plt.show()


