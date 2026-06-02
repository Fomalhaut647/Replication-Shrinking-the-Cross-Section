"""图2 — L1-L2 双惩罚 OOS R² 等高线(复现论文 Figure 3a/3b,50 anomaly)

论文标志性图。两个 panel:
  (a) 原始 50 anomaly;  (b) 其主成分(PC)。
横轴 κ(L2 强度,右弱左强),纵轴非零系数个数(L1 稀疏度,上稠下稀,对数),色=OOS R²(暖=高)。

结论(对照论文):
  (a) 高 R² 区是一条近**垂直**的带(需要相当强的 L2,但几乎不能稀疏)——50 anomaly 之间
      冗余很少,沿纵轴往下(强制稀疏)R² 急剧恶化 -> characteristics-sparse SDF 不成立。
  (b) 高 R² 区延伸到图**底部**(很少几个 PC 即可)——PC 空间存在稀疏表示。
两图对比是论文核心:稀疏性在 PC 空间存在、在原始特征空间几乎不存在。
"""
import numpy as np
import scs_plot as P


def _regular_grid(NZ, R2, N):
    """把每个 κ 列的 (非零数 -> OOS R²) 插值到规则 y 网格 1..N,用于平滑等高线。"""
    y_grid = np.arange(1, N + 1)
    nk = NZ.shape[0]
    Z = np.empty((N, nk))
    for i in range(nk):
        uniq = {}
        for z, val in zip(NZ[i], R2[i]):
            uniq[int(z)] = max(uniq.get(int(z), -1e9), float(val))  # 同非零数取最优
        xs = np.array(sorted(uniq))
        ys = np.array([uniq[z] for z in xs])
        Z[:, i] = np.interp(y_grid, xs, ys)   # 端点外推为平台
    return y_grid, Z


def plot(daily=True):
    P.setup()
    import matplotlib.pyplot as plt
    g_char = np.load('outputs/grid_char.npz')
    g_pc = np.load('outputs/grid_pc.npz')
    N = int(g_char['N'])
    levels = np.linspace(-0.1, 0.3, 25)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    titles = ['(a) Raw 50 anomaly portfolios', '(b) PCs of 50 anomaly portfolios']
    cf = None
    for ax, g, title in zip(axes, [g_char, g_pc], titles):
        y_grid, Z = _regular_grid(g['NZ'], g['R2'], N)
        Zc = np.clip(Z, -0.1, 0.3)
        cf = ax.contourf(g['kappas'], y_grid, Zc, levels=levels, cmap='viridis', extend='both')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Root Expected SR$^2$ (prior), $\\kappa$', fontweight='bold')
        ax.set_title(title, fontsize=10)
        ax.set_ylim(1, N)
        ax.grid(False)
    axes[0].set_ylabel('Number of non-zero coefficients', fontweight='bold')
    cbar = fig.colorbar(cf, ax=axes, fraction=0.046, pad=0.02)
    cbar.set_label('OOS cross-sectional $R^2$', fontweight='bold')
    fig.suptitle('Fig 2 — Dual-penalty (L1+L2) OOS $R^2$ (50 anomalies)',
                 fontsize=12, fontweight='bold')
    return P.save(fig, 'fig2_dual_penalty_contour')


if __name__ == '__main__':
    path = plot()
    # 简单数值自检:raw 高分区应在高非零数,PC 高分区应能下探到低非零数
    g_char = np.load('outputs/grid_char.npz')
    g_pc = np.load('outputs/grid_pc.npz')
    def best_nz_at_maxR2(g):
        i, j = np.unravel_index(np.argmax(g['R2']), g['R2'].shape)
        return int(g['NZ'][i, j])
    print(f"[图2] raw 最优点非零数={best_nz_at_maxR2(g_char)} (应接近 50,无稀疏)")
    print(f"      PC  最优点非零数={best_nz_at_maxR2(g_pc)} (可较小)")
    # PC 在低非零数(<=10)的最高 OOS R²
    pc_lo = g_pc['R2'][g_pc['NZ'] <= 10]
    ch_lo = g_char['R2'][g_char['NZ'] <= 10]
    print(f"      非零数<=10 时最高 OOS R²:  PC={pc_lo.max():.3f}  vs  char={ch_lo.max():.3f}")
    print(f"  saved: {path}")
