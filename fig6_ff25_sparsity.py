"""图6(扩展 R1)— FF25 上的稀疏 vs 稠密:把 KNS 管线套到经典 25 个 size/BM 组合

动机:论文 4.1 节用 FF25 做低维预备分析。本图把与 50 anomaly **完全一致**的 KNS 口径
(de-market + de-vol + 收缩 Σ + PC 旋转 + 3-fold OOS CSR² + 双惩罚稀疏前沿)套到经典
FF25 测试资产上,样本窗对齐 1963-07~2017-12,考察"稀疏性在 PC 空间 vs 特征空间"的结论
在另一个、低维且结构清晰的横截面上如何变化。

诚实预期:FF25 维度低、主要被 size/value 少数因子张成,故 characteristics-sparse 的代价
远小于 50 anomaly 的情形 —— 这恰好凸显论文的点在高维 anomaly zoo 里才尖锐。出图与打印
关键数值后由报告据实解读(无论差距大小)。
"""
import numpy as np
import scs_core as C
from scs_data import prepared_ff25, prepared
from dual_penalty import scan, sparsity_frontier
import scs_plot as P


def compute(daily=True):
    D = prepared_ff25(daily=daily)
    # 双惩罚网格扫描:与 fig2/fig4 同一函数、同一口径,仅数据换为 FF25
    res_c = scan(space='char', daily=daily, D=D)
    res_p = scan(space='pc', daily=daily, D=D)
    np.savez('outputs/grid_ff25_char.npz', **res_c)
    np.savez('outputs/grid_ff25_pc.npz', **res_p)
    ks_c, best_c, se_c = sparsity_frontier(res_c)
    ks_p, best_p, se_p = sparsity_frontier(res_p)
    # 累计方差(特征值)对比:FF25 vs 50-anomaly,展示 FF25 维度更低(论文 4.1 的低维 warm-up)
    d_ff = np.sort(D['d'])[::-1]; cum_ff = np.cumsum(d_ff) / d_ff.sum()
    d_an = np.sort(prepared(daily=daily)['d'])[::-1]; cum_an = np.cumsum(d_an) / d_an.sum()
    return dict(ks_c=ks_c, best_c=best_c, se_c=se_c,
                ks_p=ks_p, best_p=best_p, se_p=se_p,
                cum_ff=cum_ff, cum_an=cum_an,
                res_c=res_c, res_p=res_p, D=D)


def plot(res):
    P.setup()
    import matplotlib.pyplot as plt
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 4.7))
    # (a) 累计方差:FF25 vs 50-anomaly
    axA.plot(np.arange(1, len(res['cum_ff']) + 1), res['cum_ff'], 'o-', color='tab:red',
             lw=2.0, ms=4, label='FF25 (25 size/BM)')
    axA.plot(np.arange(1, len(res['cum_an']) + 1), res['cum_an'], 's-', color='tab:blue',
             lw=1.6, ms=3, label='50 anomalies (main)')
    axA.axhline(0.9, color='k', ls=':', lw=0.8, alpha=0.6)
    axA.set_xlabel('Number of PCs', fontweight='bold')
    axA.set_ylabel('Cumulative variance share', fontweight='bold')
    axA.set_xlim(0, 26)
    axA.legend(loc='lower right', fontsize=9)
    axA.set_title(f"(a) FF25 is lower-dimensional — top-3 PCs = {res['cum_ff'][2]:.0%} of variance",
                  fontsize=9.5)
    # (b) OOS 稀疏 vs 稠密
    mc = np.isfinite(res['best_c']); mp = np.isfinite(res['best_p'])
    axB.plot(res['ks_c'][mc], res['best_c'][mc], '-', color='tab:blue', lw=2.0,
             label='Characteristics (FF25)')
    axB.plot(res['ks_p'][mp], res['best_p'][mp], '--', color='tab:red', lw=2.0, label='PCs (FF25)')
    axB.set_xscale('log')
    axB.set_xlabel('Number of factors in the SDF', fontweight='bold')
    axB.set_ylabel('OOS Cross-sectional $R^2$', fontweight='bold')
    axB.axhline(0.0, color='k', lw=0.7, alpha=0.5)
    axB.legend(loc='upper left', fontsize=9)
    axB.set_title('(b) OOS $R^2$ $\\approx$ 0 (de-marketed residual premia time-unstable)',
                  fontsize=9.5)
    return P.save(fig, 'fig6_ff25_sparsity')


if __name__ == '__main__':
    res = compute(daily=True)
    np.savez('outputs/fig6_data.npz',
             ks_c=res['ks_c'], best_c=res['best_c'], se_c=res['se_c'],
             ks_p=res['ks_p'], best_p=res['best_p'], se_p=res['se_p'])
    path = plot(res)
    D = res['D']
    d0, dN = str(D['dates'].iloc[0])[:10], str(D['dates'].iloc[-1])[:10]
    print(f"[图6/FF25]  T={D['T']}  N={D['N']}  样本 {d0} ~ {dN}")
    print(f"  网格 OOS R²: char∈[{res['res_c']['R2'].min():.3f},{res['res_c']['R2'].max():.3f}]"
          f"  pc∈[{res['res_p']['R2'].min():.3f},{res['res_p']['R2'].max():.3f}]")
    print("  frontier(每个因子数扫遍所有 L2 强度的最大 OOS R²):")
    for k in [1, 2, 3, 5, 10, 25]:
        bc = res['best_c'][k-1] if k <= len(res['best_c']) else np.nan
        bp = res['best_p'][k-1] if k <= len(res['best_p']) else np.nan
        print(f"   {k:2d} factors:  char OOS R²={bc:6.3f}   PC OOS R²={bp:6.3f}")
    print(f"  saved: {path}")
