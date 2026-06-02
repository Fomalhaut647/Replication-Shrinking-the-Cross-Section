"""dual_penalty.py — 双惩罚(elastic net)网格扫描,供图2(等高线)与图4(稀疏前沿)共用

对一张 (κ, γ1) 网格:
  - κ(横轴)经 kappa_to_gamma 映射为 L2 强度 γ2;κ 大 -> L2 弱。
  - γ1(L1 强度)从 γ1_max(全归零)对数降到 γ1_max·1e-3(几乎不稀疏)。
每个网格点记录:
  - NZ  : 全样本 elastic net 解的非零系数个数(= 图2/4 的"因子数"轴)
  - R2  : 3-fold OOS 横截面 R²(式30,与图1同一把尺子)
  - R2se: fold 间标准误

两个特征空间:
  space='char' 原始 50 anomaly;  space='pc' 其主成分(全样本 PC,论文 Figure 3b/4b 口径)。
warm-start 沿 γ1 path(稀疏->稠密)大幅加速坐标下降。
"""
import numpy as np
import scs_core as C
from scs_data import prepared


def scan(space='char', daily=True, nkappa=26, ngamma1=42, kappa_lo=0.05, kappa_hi=3.0, D=None):
    if D is None:
        D = prepared(daily=daily)
    T, freq, N, K = D['T'], D['freq'], D['N'], D['K']
    if space == 'char':
        Sig, mu, folds = D['Sigma'], D['mu'], D['folds']
    elif space == 'pc':
        Pm = D['P']
        Sig = C.regcov(Pm)
        mu = Pm.mean(0)
        folds = C.precompute_folds(Pm, K)
    else:
        raise ValueError(space)

    g1max = C.enet_lambda1_max(mu)
    g1_grid = np.logspace(np.log10(g1max), np.log10(g1max * 1e-3), ngamma1)  # 降序
    kappas = np.logspace(np.log10(kappa_lo), np.log10(kappa_hi), nkappa)
    f = 1.0 / (1.0 - 1.0 / K)

    NZ = np.zeros((nkappa, ngamma1), int)
    R2 = np.zeros((nkappa, ngamma1))
    R2se = np.zeros((nkappa, ngamma1))

    for i, k in enumerate(kappas):
        g2 = C.kappa_to_gamma(k, Sig, T, freq)
        # 全样本:非零系数个数(沿 γ1 warm)
        B = C.enet_path(Sig, mu, g2, g1_grid, warm=True)
        NZ[i] = (np.abs(B) > 1e-8).sum(1)
        # CV:每 fold 沿 γ1 warm-start
        fold_oos = np.zeros((K, ngamma1))
        for fi, (Xtr, ytr, Xte, yte) in enumerate(folds):
            b0 = None
            for j, g1 in enumerate(g1_grid):
                # 仅 L2(g2)随折缩放;L1(g1)不缩放,使 fold 模型稀疏度与全样本 y 轴标注一致
                b = C.enet_coef(Xtr, ytr, g2 * f, g1, b0=b0)
                b0 = b
                fold_oos[fi, j] = C.csr2(Xte @ b, yte)
        R2[i] = fold_oos.mean(0)
        R2se[i] = fold_oos.std(0) / np.sqrt(K)
    return dict(kappas=kappas, g1_grid=g1_grid, NZ=NZ, R2=R2, R2se=R2se, N=N, space=space)


def sparsity_frontier(scan_res, monotone=True):
    """从网格提取"每个因子数 k 的最大 OOS R²(扫遍所有 L2 强度)" —— 论文 Figure 4b 的 ridge。

    monotone=True:对均值线施加 running-max 单调包络。语义="用至多 k 个因子的最优模型"
    随 k 单调不减(更多因子可退化到更稀疏解),消除 L1 在相关 char 上 NZ 跳变造成的锯齿,
    与论文 Figure 4b 的平滑上升一致。se 带在缺失 k 处做前向填充对齐。
    """
    NZ, R2, R2se, N = scan_res['NZ'], scan_res['R2'], scan_res['R2se'], scan_res['N']
    ks = np.arange(1, N + 1)
    best = np.full(N, np.nan)
    best_se = np.full(N, np.nan)
    for idx, k in enumerate(ks):
        mask = NZ == k
        if mask.any():
            vals = R2[mask]
            jbest = int(np.argmax(vals))
            best[idx] = vals[jbest]
            best_se[idx] = R2se[mask][jbest]
    if monotone:
        # 前向填充缺失 k(用上一个可用值),再做 running max 得单调包络
        last = np.nan
        for i in range(N):
            if np.isfinite(best[i]):
                last = best[i]
            elif np.isfinite(last):
                best[i] = last
                best_se[i] = best_se[i - 1] if i > 0 else np.nan
        filled = np.where(np.isfinite(best), best, -np.inf)
        cummax = np.maximum.accumulate(filled)
        # 被 running-max 抬升的点,其 se 沿用产生该 max 的点
        run_se = np.full(N, np.nan)
        cur_se = np.nan
        cur_val = -np.inf
        for i in range(N):
            if filled[i] >= cur_val:
                cur_val = filled[i]
                cur_se = best_se[i]
            run_se[i] = cur_se
        best = np.where(cummax > -np.inf, cummax, np.nan)
        best_se = run_se
    return ks, best, best_se


if __name__ == '__main__':
    import sys
    daily = True
    D = prepared(daily=daily)
    for space in ['char', 'pc']:
        print(f"扫描 space={space} ...", flush=True)
        res = scan(space=space, daily=daily, D=D, nkappa=32, ngamma1=60)
        np.savez(f'outputs/grid_{space}.npz', **res)
        ks, best, se = sparsity_frontier(res)
        imax = int(np.nanargmax(res['R2']))
        flat = res['R2'].ravel()
        print(f"  网格 OOS R² 范围 [{flat.min():.3f}, {flat.max():.3f}]")
        # 关键 frontier 点
        for k in [1, 2, 4, 10, 25, 50]:
            if k <= len(best):
                print(f"  {space} frontier: {k:2d} factors -> OOS R²={best[k-1]:.3f}")
    print("done.")
