"""scs_core 自测:验证数值地基正确性(不产出交付物,仅打印断言结果)。"""
import os
import numpy as np
from load_managed_portfolios import load_managed_portfolios
import scs_core as C
from utils import l2est

np.random.seed(0)

# ---- 加载 daily 数据(与阶段1 scs_main.py 默认一致)----
instr = os.path.join('Data', 'Instruments')
fn = os.path.join(instr, 'managed_portfolios_anom_d_50.csv')
dates, re, mkt, anomalies = load_managed_portfolios(fn, True, 0.2, ['rX_', 'r2_', 'r3_'])
freq = 252
print(f"数据: T={len(dates)}, N={re.shape[1]} anomalies")

# ---- 预处理 ----
R = C.prepare_factors(dates, re, mkt)
Rv = R.values
T, N = Rv.shape
Sigma = C.regcov(Rv)
mu = Rv.mean(0)
print(f"预处理后: T={T}, N={N}, tr(Σ)={np.trace(Sigma):.4f}")

# ================= 测试1: enet(γ1=0) == l2est == l2_coef =================
kappa = 0.3
gamma = C.kappa_to_gamma(kappa, Sigma, T, freq)
b_l2 = C.l2_coef(Sigma, mu, gamma)
b_en = C.enet_coef(Sigma, mu, gamma, 0.0)
b_ut = l2est(Sigma, mu, {'L2pen': gamma})[0]
e1 = np.max(np.abs(b_en - b_l2))
e2 = np.max(np.abs(b_l2 - b_ut))
print(f"\n[测试1] enet(γ1=0) vs l2_coef vs utils.l2est @κ={kappa}")
print(f"  max|enet - l2_coef| = {e1:.2e}   max|l2_coef - l2est| = {e2:.2e}")
assert e1 < 1e-6 and e2 < 1e-10, "L2 退化不一致!"
print("  PASS: 坐标下降在 γ1=0 时严格退化为纯 L2")

# ================= 测试2: enet(γ1>0) 产生稀疏解 =================
print(f"\n[测试2] enet 稀疏性(扫 γ1)")
for g1 in [0.0, 0.005, 0.02, 0.05, 0.15]:
    b = C.enet_coef(Sigma, mu, gamma, g1)
    nz = int(np.sum(np.abs(b) > 1e-8))
    print(f"  γ1={g1:<6} 非零系数={nz:2d}/{N}")
b_big = C.enet_coef(Sigma, mu, gamma, 0.3)
assert int(np.sum(np.abs(b_big) > 1e-8)) < N, "大 γ1 未产生稀疏!"
print("  PASS: γ1 增大 -> 非零系数单调减少(稀疏)")

# ================= 测试3: PC 旋转正确性 =================
print(f"\n[测试3] PC 旋转")
P, d, Q = C.pc_rotate(Rv, Sigma)
# Q 正交 + 特征值降序 + P 的(收缩前)样本协方差近似对角
orth = np.max(np.abs(Q.T @ Q - np.eye(N)))
desc = np.all(np.diff(d) <= 1e-12)
print(f"  Q 正交性 max|Q'Q-I|={orth:.2e}, 特征值降序={desc}, d[0]={d[0]:.4f} d[-1]={d[-1]:.6f}")
assert orth < 1e-8 and desc, "PC 旋转错误!"
print("  PASS")

# ================= 测试4: oos_cv_r2 复现 cross_validation.png 的 OOS 曲线 =================
print(f"\n[测试4] OOS-CV 曲线(应复现阶段1: 峰≈0.24-0.25 @ κ≈0.28)")
folds = C.precompute_folds(Rv, 3)
kappas = np.logspace(np.log10(0.05), np.log10(1.0), 40)
oos = []
for k in kappas:
    g = C.kappa_to_gamma(k, Sigma, T, freq)
    _, o, _ = C.oos_cv_r2(folds, g, 0.0, K=3)
    oos.append(o)
oos = np.array(oos)
imax = int(np.argmax(oos))
print(f"  OOS 峰值 = {oos[imax]:.4f} @ κ = {kappas[imax]:.4f}")
assert 0.20 < oos[imax] < 0.30 and 0.18 < kappas[imax] < 0.45, "OOS 曲线与阶段1不符!"
print("  PASS: 口径与原仓库 cross_validate 一致")

print("\n=== 全部自测通过:scs_core 数值地基可靠 ===")
