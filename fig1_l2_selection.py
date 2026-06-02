"""图1 — L2 模型选择(复现论文 Figure 4a,50 anomaly)

内容:横轴 κ(root expected SR²,先验强度;κ 越大 -> L2 惩罚越弱),纵轴横截面 R²。
  - In-sample(虚线蓝):随 κ 增大单调上升 -> 全样本拟合可被噪声无限抬高。
  - OOS CV(实线橙):先升后降的驼峰,峰值即数据驱动选出的最优收缩 κ*。
  - ±1 s.e.(点线橙):3-fold CV 的 fold 间标准误带。
  - P&S level shrinkage(点划黄):Pástor-Stambaugh(2000) η=1 先验,对所有 PC **等比例**
    收缩(不做 relative twist)。论文据此论证"相对收缩(downweight 小 PC)"才是关键,
    仅 level 收缩的 OOS R² 远低于我们的 η=2 估计。

为什么是这个形状:in-sample R² 把均值的样本噪声也拟合进去,故无收缩时虚高;OOS 下噪声
不再复现,过弱的收缩(大 κ)反而过拟合,故 OOS 出现内点最优 —— 这正是论文"需要收缩"的核心证据。
"""
import numpy as np
import scs_core as C
from scs_data import prepared
import scs_plot as P


def pastor_stambaugh_oos(folds, gamma, N, K=3):
    """P&S(2000) level shrinkage 的 OOS CSR²(纯 level、无 relative twist)。

    实现:对所有 PC 施加**同一个** level 收缩因子 c = d̄/(d̄+γ)(d̄=tr Σ/N),
    等价 raw 空间 b = c·Σ⁻¹μ;区别于本文 η=2 的 relative 因子 d_j/(d_j+γ)(对小 PC 收缩更狠)。
    注:这是对 Pástor-Stambaugh level 收缩的**合理重参数化**(论文脚注15亦指出其 x 轴不再等于 κ),
    并非字面 η=1 后验;但它确实是"等比例、不 twist"的 level 收缩,足以说明 level 远逊于 relative。
    """
    f = 1.0 / (1.0 - 1.0 / K)
    g = gamma * f
    oos = []
    for (Xtr, ytr, Xte, yte) in folds:
        dbar = np.trace(Xtr) / N
        c = dbar / (dbar + g)                       # level 收缩因子(标量,所有 PC 相同)
        b = c * np.linalg.pinv(Xtr) @ ytr
        oos.append(C.csr2(Xte @ b, yte))
    return float(np.mean(oos))


def compute(daily=True, ngrid=60):
    D = prepared(daily=daily)
    Sigma, mu, T, freq, folds = D['Sigma'], D['mu'], D['T'], D['freq'], D['folds']
    kappas = np.logspace(np.log10(0.02), np.log10(1.2), ngrid)
    IS = np.empty(ngrid); OOS = np.empty(ngrid); SE = np.empty(ngrid); PS = np.empty(ngrid)
    for i, k in enumerate(kappas):
        g = C.kappa_to_gamma(k, Sigma, T, freq)
        IS[i], OOS[i], SE[i] = C.oos_cv_r2(folds, g, 0.0, K=D['K'])
        PS[i] = pastor_stambaugh_oos(folds, g, D['N'], K=D['K'])
    return dict(kappas=kappas, IS=IS, OOS=OOS, SE=SE, PS=PS,
                kstar=float(kappas[np.argmax(OOS)]), r2star=float(np.max(OOS)))


def plot(res):
    P.setup()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    k = res['kappas']
    ax.plot(k, res['IS'], '--', color='tab:blue', lw=1.8, label='In-sample')
    ax.plot(k, res['OOS'], '-', color='tab:orange', lw=2.0, label='OOS CV')
    ax.plot(k, res['OOS'] + res['SE'], ':', color='tab:orange', lw=1.0)
    ax.plot(k, res['OOS'] - res['SE'], ':', color='tab:orange', lw=1.0,
            label='OOS CV $\\pm$ 1 s.e.')
    ax.plot(k, res['PS'], '-.', color='goldenrod', lw=1.6, label='P&S level shrinkage ($\\eta=1$)')
    ax.axvline(res['kstar'], color='k', ls=':', lw=1.0, alpha=0.6)
    ax.set_xscale('log')
    ax.set_xlabel('Root Expected SR$^2$ (prior), $\\kappa$', fontweight='bold')
    ax.set_ylabel('IS / OOS Cross-sectional $R^2$', fontweight='bold')
    ax.set_ylim(0, 0.62)   # 固定纵轴(对齐论文 Fig.4a;In-sample 线会冲出顶部)
    ax.set_xlim(k.min(), k.max())
    ax.legend(loc='upper left')
    ax.set_title(f"L2 model selection (50 anomalies)  —  $\\kappa^*$={res['kstar']:.3f}, "
                 f"OOS $R^2$={res['r2star']:.3f}", fontsize=10)
    return P.save(fig, 'fig1_l2_selection')


if __name__ == '__main__':
    res = compute(daily=True)
    np.savez('outputs/fig1_data.npz', **res)
    path = plot(res)
    print(f"[图1] OOS 峰值 R²={res['r2star']:.4f} @ κ*={res['kstar']:.4f}")
    print(f"  In-sample @κ=1.2: {res['IS'][-1]:.4f}  (应远高于 OOS,体现过拟合)")
    print(f"  P&S level shrinkage 峰值: {np.max(res['PS']):.4f}  (应 << OOS 峰值)")
    print(f"  saved: {path}")
