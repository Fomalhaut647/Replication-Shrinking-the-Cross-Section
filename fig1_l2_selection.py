"""图1 — L2 模型选择(复现论文 Figure 4a,50 anomaly)+ 扩展 R4(字面 η=1 P&S)

内容:横轴 κ(root expected SR²,先验强度;κ 越大 -> L2 惩罚 γ 越弱),纵轴横截面 R²。
  - In-sample(虚线蓝,η=2):随 κ 增大单调上升 -> 全样本拟合可被噪声无限抬高。
  - OOS CV(实线橙,η=2 relative):先升后降的驼峰,峰值即数据驱动选出的最优收缩 κ*。
  - ±1 s.e.(点线橙):3-fold CV 的 fold 间标准误带。
  - **字面 η=1 level shrinkage(点划黄,扩展 R4)**:Pástor-Stambaugh(2000) 的 η=1 后验,
    b_{P,j}=μ_{P,j}/(d_j(1+γ)),即对所有 PC 的 OLS 系数**等比例**收缩。与本文 η=2 的 relative
    收缩 d_j/(d_j+γ)(对低方差 PC 收缩更狠)对比。

R4 要点(相对旧实现的改进):旧版用 level 因子 c=d̄/(d̄+γ) 的**重参数化**;本版用**字面 η=1 后验**
(论文式18-24 一般化 b_{P,j}=μ_{P,j}/(d_j+γ·d_j^{2-η}),η=1)。两种收缩共用同一 γ↔κ 映射故同轴,
但各自最优惩罚相差约两个数量级(论文脚注15:η=1 下 κ 的经济含义与 η=2 不同),驼峰自然分居左右。
诚实结论:level 收缩调到最优也仅 OOS≈0.06,远逊于 relative 收缩的 ≈0.25 —— 这正是论文核心证据。

为什么是这个形状:in-sample R² 把均值的样本噪声也拟合进去,故无收缩时虚高;OOS 下噪声不再复现,
过弱的收缩(大 κ)反而过拟合,故 OOS 出现内点最优 —— 这正是论文"需要(相对)收缩"的核心证据。
"""
import numpy as np
import scs_core as C
from scs_data import prepared
import scs_plot as P


def compute(daily=True, ngrid=80):
    D = prepared(daily=daily)
    Sigma, mu, T, freq, folds, K = D['Sigma'], D['mu'], D['T'], D['freq'], D['folds'], D['K']
    folds_eta = C.precompute_folds_eta(D['Rv'], K)      # PC 空间、每折 train-only 旋转(R2 口径)
    # κ 网格向左延伸到 0.001,以同时容纳 η=2 与 η=1 两个驼峰(后者最优在 κ≈0.003)
    kappas = np.logspace(np.log10(0.001), np.log10(1.2), ngrid)
    IS = np.empty(ngrid); OOS = np.empty(ngrid); SE = np.empty(ngrid); PS = np.empty(ngrid)
    for i, k in enumerate(kappas):
        g = C.kappa_to_gamma(k, Sigma, T, freq)
        IS[i], OOS[i], SE[i] = C.oos_cv_r2(folds, g, 0.0, K)        # η=2 relative(主方法)
        PS[i] = C.eta_oos(folds_eta, g, eta=1.0, K=K)[0]           # η=1 字面 level(R4)
    return dict(kappas=kappas, IS=IS, OOS=OOS, SE=SE, PS=PS,
                kstar=float(kappas[np.argmax(OOS)]), r2star=float(np.max(OOS)),
                ps_best=float(np.max(PS)), ps_kappa=float(kappas[np.argmax(PS)]))


def plot(res):
    P.setup()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    k = res['kappas']
    ax.plot(k, res['IS'], '--', color='tab:blue', lw=1.8, label='In-sample ($\\eta=2$)')
    ax.plot(k, res['OOS'], '-', color='tab:orange', lw=2.0,
            label='OOS CV — relative shrinkage ($\\eta=2$, KNS)')
    ax.plot(k, res['OOS'] + res['SE'], ':', color='tab:orange', lw=1.0)
    ax.plot(k, res['OOS'] - res['SE'], ':', color='tab:orange', lw=1.0, label='OOS $\\pm$ 1 s.e.')
    ax.plot(k, res['PS'], '-.', color='goldenrod', lw=1.8,
            label='OOS — level shrinkage ($\\eta=1$, P\\&S)')
    ax.axvline(res['kstar'], color='k', ls=':', lw=1.0, alpha=0.6)
    # 标注两个峰
    ax.annotate(f"relative peak\n{res['r2star']:.3f}", xy=(res['kstar'], res['r2star']),
                xytext=(res['kstar'] * 1.1, res['r2star'] + 0.07), fontsize=8.5,
                arrowprops=dict(arrowstyle='->', lw=0.8))
    ax.annotate(f"level peak\n{res['ps_best']:.3f}", xy=(res['ps_kappa'], res['ps_best']),
                xytext=(res['ps_kappa'] * 0.9, res['ps_best'] + 0.10), fontsize=8.5,
                ha='right', arrowprops=dict(arrowstyle='->', lw=0.8))
    ax.set_xscale('log')
    ax.set_xlabel('Root Expected SR$^2$ (prior), $\\kappa$', fontweight='bold')
    ax.set_ylabel('IS / OOS Cross-sectional $R^2$', fontweight='bold')
    ax.set_ylim(0, 0.62)   # 固定纵轴(对齐论文 Fig.4a;In-sample 会冲顶,η=1 灾难段在 0 下自然 clip)
    ax.set_xlim(k.min(), k.max())
    ax.legend(loc='upper left', fontsize=8.5)
    ax.set_title(f"L2 model selection (50 anomalies)  —  $\\kappa^*$={res['kstar']:.3f}, "
                 f"OOS $R^2$={res['r2star']:.3f}  vs  level best {res['ps_best']:.3f}", fontsize=9.5)
    return P.save(fig, 'fig1_l2_selection')


if __name__ == '__main__':
    res = compute(daily=True)
    np.savez('outputs/fig1_data.npz', **res)
    path = plot(res)
    print(f"[图1] η=2 relative OOS 峰值 R²={res['r2star']:.4f} @ κ*={res['kstar']:.4f}")
    print(f"  In-sample @κ={res['kappas'][-1]:.2f}: {res['IS'][-1]:.4f}  (远高于 OOS,体现过拟合)")
    print(f"  字面 η=1 level(P&S) 最优 OOS={res['ps_best']:.4f} @ κ={res['ps_kappa']:.4f}  (<< relative 峰值)")
    print(f"  saved: {path}")
