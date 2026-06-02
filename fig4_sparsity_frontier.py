"""图4 — 稀疏 vs 稠密:OOS R² 随因子数变化(复现论文 Figure 4b,50 anomaly)

横轴:进入 SDF 的因子数(非零系数个数,对数);纵轴:该稀疏度下扫遍所有 L2 强度能达到的
最大 OOS 横截面 R²(即沿图2等高线的"山脊")。两条线:
  - Characteristics(实线蓝):用原始 anomaly 因子。慢升 —— 要接近 50 个才达到最大,
    说明 characteristics-sparse SDF 表现差。
  - PCs(虚线红):用主成分。快升 —— 4 个 PC 已达约 2/3,10 个 PC 近最大,
    说明 PC-sparse SDF 表现好。
点线为 -1 s.e. 带。这是论文"稀疏性在 PC 空间存在、在特征空间不存在"的最直接证据,
也直接回答 Task.md 图4"稀疏 vs 稠密 OOS 表现对比"。

附:同时给出两种空间的样本外 **Sharpe ratio** 对比(Task.md 明确要 Sharpe),口径为
最优 κ* 下 SDF 隐含的 MVE 组合在 3-fold OOS 拼接收益上的年化 Sharpe。
"""
import numpy as np
import scs_core as C
from scs_data import prepared
from dual_penalty import sparsity_frontier
import scs_plot as P


def oos_mve_sharpe(folds, gamma2, freq, gamma1=0.0, K=3, space_returns=None):
    """最优收缩下 SDF 隐含 MVE 组合的 OOS 年化 Sharpe(拼接各 fold 的留出期收益)。

    每 fold:用训练矩估 b,在留出块上构造 MVE 组合收益 r_te·b;拼接后算年化 Sharpe。
    space_returns: list[r_te(ndarray)] 与 folds 对齐的留出期原始收益(用于组合收益)。
    """
    f = 1.0 / (1.0 - 1.0 / K)
    g2, g1 = gamma2 * f, gamma1 * f
    seg = []
    for (Xtr, ytr, _, _), r_te in zip(folds, space_returns):
        b = C.enet_coef(Xtr, ytr, g2, g1) if g1 > 0 else C.l2_coef(Xtr, ytr, g2)
        seg.append(r_te @ b)
    r = np.concatenate(seg)
    return float(np.mean(r) / np.std(r) * np.sqrt(freq))


def compute(daily=True):
    D = prepared(daily=daily)
    g_char = np.load('outputs/grid_char.npz')
    g_pc = np.load('outputs/grid_pc.npz')
    ks_c, best_c, se_c = sparsity_frontier(dict(NZ=g_char['NZ'], R2=g_char['R2'],
                                                R2se=g_char['R2se'], N=int(g_char['N'])))
    ks_p, best_p, se_p = sparsity_frontier(dict(NZ=g_pc['NZ'], R2=g_pc['R2'],
                                                R2se=g_pc['R2se'], N=int(g_pc['N'])))
    return dict(ks_c=ks_c, best_c=best_c, se_c=se_c,
                ks_p=ks_p, best_p=best_p, se_p=se_p, D=D)


def plot(res):
    P.setup()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    mc = np.isfinite(res['best_c'])
    mp = np.isfinite(res['best_p'])
    ax.plot(res['ks_c'][mc], res['best_c'][mc], '-', color='tab:blue', lw=2.0, label='Characteristics')
    ax.plot(res['ks_c'][mc], (res['best_c'] - res['se_c'])[mc], ':', color='tab:blue', lw=1.0)
    ax.plot(res['ks_p'][mp], res['best_p'][mp], '--', color='tab:red', lw=2.0, label='PCs')
    ax.plot(res['ks_p'][mp], (res['best_p'] - res['se_p'])[mp], ':', color='tab:red', lw=1.0)
    ax.set_xscale('log')
    ax.set_xlabel('Number of factors in the SDF', fontweight='bold')
    ax.set_ylabel('OOS Cross-sectional $R^2$', fontweight='bold')
    ax.set_ylim(0, max(0.32, 1.05 * np.nanmax(res['best_p'])))
    ax.legend(loc='lower right')
    ax.set_title('Fig 4 — Sparse vs dense: OOS $R^2$ by number of factors (50 anomalies)',
                 fontsize=10)
    return P.save(fig, 'fig4_sparsity_frontier')


if __name__ == '__main__':
    res = compute(daily=True)
    np.savez('outputs/fig4_data.npz', ks_c=res['ks_c'], best_c=res['best_c'], se_c=res['se_c'],
             ks_p=res['ks_p'], best_p=res['best_p'], se_p=res['se_p'])
    path = plot(res)

    # ---- 稀疏(纯L1 in char) vs 稠密(L2) 的 OOS Sharpe 对比 ----
    D = res['D']
    Sigma, mu, T, freq, K = D['Sigma'], D['mu'], D['T'], D['freq'], D['K']
    folds = D['folds']
    # 各 fold 留出期的原始(char)收益,用于构造 MVE 组合收益
    Rv = D['Rv']
    te_returns = [Rv[te] for te in C.cv_partition_contiguous(T, K)]
    kstar = float(np.load('outputs/fig1_data.npz')['kstar'])
    g_dense = C.kappa_to_gamma(kstar, Sigma, T, freq)
    sharpe_dense = oos_mve_sharpe(folds, g_dense, freq, 0.0, K, te_returns)
    # 稀疏:在 char 空间用较强 L1 限制到 ~5 个因子,L2 仍取 κ*
    g1 = 0.5 * C.enet_lambda1_max(mu)
    nz = int((np.abs(C.enet_coef(Sigma, mu, g_dense, g1)) > 1e-8).sum())
    sharpe_sparse = oos_mve_sharpe(folds, g_dense, freq, g1, K, te_returns)

    print(f"[图4] frontier:")
    for k in [1, 2, 4, 10, 25, 50]:
        bc = res['best_c'][k-1] if k <= len(res['best_c']) else np.nan
        bp = res['best_p'][k-1] if k <= len(res['best_p']) else np.nan
        print(f"   {k:2d} factors:  char OOS R²={bc:6.3f}   PC OOS R²={bp:6.3f}")
    print(f"  OOS MVE Sharpe(年化):  稠密 L2(50因子)={sharpe_dense:.3f}   "
          f"稀疏 L1(~{nz}因子)={sharpe_sparse:.3f}")
    print(f"  saved: {path}")
