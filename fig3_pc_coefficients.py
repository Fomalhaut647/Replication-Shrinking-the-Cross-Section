"""图3 — SDF 系数在 PC 空间的分布(复现论文 Table 1b + 式24)

论文核心论点:SDF 的定价信息**不集中在前几个主成分**。若信息只在前 2-3 个高方差 PC,
则稀疏 PC 模型即可;但实证上中等方差 PC(PC10/11/15/19…)也带显著载荷,故需要相当多 PC。

做法:在交叉验证选出的最优 κ*(来自图1)下,把 SDF 系数旋转到 PC 空间:
  b_P,j = μ_P,j/(d_j+γ)          (式24)
  t_j   = b_P,j / sqrt((1/T)/(d_j+γ))   (式23 的标准误)
按 PC 序号(1=最高方差)展示 |t| 与 b_P。对照论文 Table 1b:|t| 最大的是 PC5,PC1,PC2,PC4,
但 PC11/PC15 等也 >1,印证信息分散。
"""
import numpy as np
import scs_core as C
from scs_data import prepared
import scs_plot as P


def compute(daily=True, kappa_star=None):
    D = prepared(daily=daily)
    Sigma, mu, T, freq = D['Sigma'], D['mu'], D['T'], D['freq']
    Pmat, d, Q = D['P'], D['d'], D['Q']
    if kappa_star is None:
        # 若未传入,用图1的数据;否则回退到论文/阶段1的 κ*≈0.28
        try:
            kappa_star = float(np.load('outputs/fig1_data.npz')['kstar'])
        except Exception:
            kappa_star = 0.28
    gamma = C.kappa_to_gamma(kappa_star, Sigma, T, freq)
    b_P, se_P, mu_P = C.pc_space_l2(Pmat, d, gamma, T)
    t_P = b_P / se_P
    order_t = np.argsort(-np.abs(t_P))           # 按 |t| 降序
    return dict(b_P=b_P, t_P=t_P, d=d, mu_P=mu_P, kappa_star=kappa_star,
                order_t=order_t, T=T)


def plot(res):
    P.setup()
    import matplotlib.pyplot as plt
    b_P, t_P, d = res['b_P'], res['t_P'], res['d']
    n = len(b_P)
    pc_idx = np.arange(1, n + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # (a) |t-stat| vs PC 序号
    ax = axes[0]
    colors = ['tab:red' if abs(t) >= 2 else 'tab:blue' for t in t_P]
    ax.bar(pc_idx, np.abs(t_P), color=colors, width=0.85)
    ax.axhline(2.0, color='k', ls='--', lw=1.0, alpha=0.7, label='|t| = 2')
    # 标注 |t| 最大的前 6 个 PC
    for j in res['order_t'][:6]:
        ax.annotate(f'PC{j+1}', (j + 1, abs(t_P[j])),
                    textcoords='offset points', xytext=(0, 4),
                    ha='center', fontsize=8, fontweight='bold')
    ax.set_xlabel('PC index (1 = highest variance)', fontweight='bold')
    ax.set_ylabel('|$t$-statistic|', fontweight='bold')
    ax.set_title('(a) $t$-stats of PC loadings: significance spreads to mid-variance PCs', fontsize=9.5)
    ax.legend(loc='upper right')

    # (b) SDF 系数 b_P vs PC 序号
    ax = axes[1]
    ax.bar(pc_idx, b_P, color='tab:purple', width=0.85)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('PC index (1 = highest variance)', fontweight='bold')
    ax.set_ylabel('SDF coefficient $b_{P,j}$', fontweight='bold')
    ax.set_title(f"(b) SDF coefficients in PC space ($\\kappa^*$={res['kappa_star']:.3f})", fontsize=10)

    fig.suptitle('Fig 3 — SDF coefficient distribution across PCs (50 anomalies)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    return P.save(fig, 'fig3_pc_coefficients')


if __name__ == '__main__':
    res = compute(daily=True)
    np.savez('outputs/fig3_data.npz', b_P=res['b_P'], t_P=res['t_P'],
             d=res['d'], kappa_star=res['kappa_star'])
    path = plot(res)
    print(f"[图3] κ*={res['kappa_star']:.4f}")
    print("  |t| 最大的前 11 个 PC(对照论文 Table 1b: PC5,PC1,PC2,PC4,PC11,PC15,PC10,PC6,PC19...):")
    for r, j in enumerate(res['order_t'][:11], 1):
        print(f"   {r:2d}. PC{j+1:<3d}  b={res['b_P'][j]:+.3f}  |t|={abs(res['t_P'][j]):.2f}")
    nsig = int(np.sum(np.abs(res['t_P']) >= 2))
    print(f"  |t|>=2 的 PC 个数: {nsig}(若信息只在前几个 PC,该数应很小)")
    print(f"  saved: {path}")
