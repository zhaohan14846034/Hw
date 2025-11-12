# -*- coding: utf-8 -*-
#14846034 曾肇瀚
"""
Created on Mon Mar 15 09:37:05 2021

@author: htchen
"""
# If this script is not run under spyder IDE, comment the following two lines.
#from IPython import get_ipython
#get_ipython().run_line_magic('reset', '-sf')

import math
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import make_interp_spline

plt.rcParams['figure.dpi'] = 144 

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

# 讀取影像檔, 並保留亮度成分
IMG_PATH = r"C:\Users\ASUS\Downloads\svd_demo1.jpg"
img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"[Info] ⚠️ 無法讀取影像：{IMG_PATH}")
    print("[Info] ▶ 改用程式自動產生的 256x256 灰階漸層測試影像。")
    h, w = 256, 256
    x = np.linspace(0, 255, w, dtype=np.float64)
    A = np.tile(x, (h, 1))  # 自動生成水平灰階漸層影像
else:
    print(f"[Info] ✅ 成功讀取影像：{IMG_PATH}")
    A = img.astype(np.float64)


# SVD of A
U, Sigma, V = mysvd(A)
VT = V.T


def compute_energy(X: np.ndarray):
    # return energy of X
    # For more details on the energy of a 2D signal, see the 
    # class notebook: 內容庫/補充說明/Energy of a 2D Signal.
    return float(la.norm(X, 'fro') ** 2)
    
    
# img_h and img_w are image's height and width, respectively
img_h, img_w = A.shape
# Compute SNR
keep_r = 201
rs = np.arange(1, keep_r)


# compute energy of A, and save it to variable Energy_A
energy_A = compute_energy(A)

# Decalre an array to save the energy of noise vs r.
# energy_N[r] is the energy of A - A_bar(sum of the first r components)
energy_N = np.zeros(keep_r) # energy_N[0]棄置不用

for r in rs:
    # A_bar is the sum of the first r comonents of SVD
    # A_bar is an approximation of A
    A_bar = U[:, 0:r] @ Sigma[0:r, 0:r] @ VT[0:r, :] 
    Noise = A - A_bar 
    energy_N[r] = compute_energy(Noise) 

# 計算snr和作圖
def main():
    
    # 1) 讀圖（灰階），若讀不到就自動產生一張 256x256 漸層測試圖
    IMG_PATH = "C:\\Users\\ASUS\\Downloads\\svd_demo1.jpg"
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        h, w = 256, 256
        x = np.linspace(0, 1, w)[None, :]
        y = np.linspace(0, 1, h)[:, None]

        # --- 加入水平與垂直漸層 ---
        grad_h = 180 * x
        grad_v = 120 * y

        # --- 多個高斯亮點模擬自然場景 ---
        yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
        gauss1 = 100 * np.exp(-3.0 * ((xx+0.5)**2 + (yy+0.3)**2))
        gauss2 = 80  * np.exp(-4.0 * ((xx-0.4)**2 + (yy-0.5)**2))
        gauss3 = 60  * np.exp(-5.0 * ((xx+0.3)**2 + (yy-0.6)**2))

        # --- 加強低頻+高頻雜訊 ---
        rng = np.random.RandomState(1)
        texture = 40 * rng.rand(h, w)                   # 高頻紋理
        noise = 15 * rng.normal(size=(h, w))            # 雜訊
        blur = cv2.GaussianBlur(texture, (9, 9), 2)     # 模糊低頻層

        # --- 組合成完整影像 ---
        A = grad_h + grad_v + gauss1 + gauss2 + gauss3 + blur + noise
        A = np.clip(A, 0, 255).astype(np.float64)
    else:
        A = img.astype(np.float64)

    # 2) SVD
    U, Sigma, V = mysvd(A)
    VT = V.T
    rank = Sigma.shape[0]
    print(f"[Info] Rank (non-zero singular values): {rank}")

    # 3) SNR vs r
    R_MAX = 200
    rank = Sigma.shape[0]
    max_r_allowed = min(rank, keep_r - 1)
    rs = np.arange(1, min(R_MAX, rank) + 1)# ✅ 不超過秩
    energy_A = compute_energy(A)
    energy_N = np.zeros(R_MAX + 1)
    snr_db   = np.zeros(R_MAX + 1)
    eps = 1e-12  # ✅ 防止除以 0

    for r in rs:
        A_bar = U[:, :r] @ Sigma[:r, :r] @ VT[:r, :]
        Noise = A - A_bar
        energy_N[r] = compute_energy(Noise)
        snr_db[r] = 10.0 * np.log10(energy_A / (energy_N[r] + eps))  # ✅ SNR 定義

    snr_db = np.nan_to_num(snr_db, nan=0.0, posinf=60.0, neginf=0.0)
    
    r_smooth = np.linspace(rs[2], rs.max(), 500)
    snr_smooth = make_interp_spline(rs[2:], snr_db[rs[2:]])(r_smooth)

    plt.figure(figsize=(6,4))
    plt.plot(r_smooth, snr_smooth, color='red', linewidth=2)
    plt.xlabel('r')
    plt.ylabel('SNR')
    plt.xlim(0, 200)
    plt.ylim(10, 50)
    plt.xticks(np.arange(0, 225, 25))
    plt.yticks(np.arange(10, 55, 5))
    plt.grid(False)
    plt.tight_layout()
    plt.show()


# --------------------------
# verify that energy_N[r] equals the sum of lambda_i, i from r+1 to i=n,
# lambda_i is the eigenvalue of A^T @ A
# write your code here
# 作圖：A[r] (SNR, dB) vs. r

# --------------------------
# verify that energy_N[r] equals the sum of lambda_i, i from r+1 to n,
# or equivalently, ||A_bar||^2 equals the sum of the first r eigenvalues.
# 其中 lambda_i 是 A^T A 的特徵值（由大到小）。
# --------------------------

# 取得 A^T A 的特徵值（遞減排序）
    lambdas_full = np.real(la.eigvalsh(A.T @ A))
    lambdas_full = np.sort(lambdas_full)[::-1]  # 由大到小
    n_eigs = lambdas_full.shape[0]

    # 驗證能量關係
    # noise_energy_from_eigs[r] = sum_{i=r}^{n-1} lambdas_full[i] (0-based index)
    noise_energy_from_eigs = np.zeros(keep_r)
    approx_energy_from_eigs = np.zeros(keep_r)

    for r in rs:
        if r <= n_eigs:
            noise_energy_from_eigs[r]  = float(np.sum(lambdas_full[r:]))        # 後半尾和
            approx_energy_from_eigs[r] = float(np.sum(lambdas_full[:r]))        # 前 r 項和
        else:
            noise_energy_from_eigs[r]  = 0.0
            approx_energy_from_eigs[r] = float(np.sum(lambdas_full))            # 全部

    # 直接用重建能量驗證 ||A_bar||^2
    approx_energy_direct = np.zeros(keep_r)
    for r in rs:
            if r <= n_eigs:
                noise_energy_from_eigs[r] = float(np.sum(lambdas_full[r:]))
                approx_energy_from_eigs[r] = float(np.sum(lambdas_full[:r]))
            if r <= max_r_allowed:
                A_bar = U[:, :r] @ Sigma[:r, :r] @ VT[:r, :]
                approx_energy_direct[r] = compute_energy(A_bar)
            else:
                approx_energy_direct[r] = approx_energy_direct[r-1]

    # 誤差評估（絕對/相對）
    valid_r = rs[rs <= min(max_r_allowed, n_eigs)]
    noise_abs_err = np.max(np.abs(energy_N[valid_r] - noise_energy_from_eigs[valid_r]))
    approx_abs_err = np.max(np.abs(approx_energy_direct[valid_r] - approx_energy_from_eigs[valid_r]))

    noise_rel_err = noise_abs_err / (np.max(noise_energy_from_eigs[valid_r]) + 1e-12)
    approx_rel_err = approx_abs_err / (np.max(approx_energy_from_eigs[valid_r]) + 1e-12)

    print(f"[Verify] max |energy_N[r] - tail_sum(lambda)| = {noise_abs_err:.6e}, rel = {noise_rel_err:.6e}")
    print(f"[Verify] max ||A_bar||^2 - sum_1^r(lambda)   = {approx_abs_err:.6e}, rel = {approx_rel_err:.6e}")
    print(f"Total energy ||A||_F^2 = {energy_A:.6f}  vs  sum(lambda_i) = {np.sum(lambdas_full):.6f}")

# ---------- 程式入口 ----------
if __name__ == "__main__":
    main()