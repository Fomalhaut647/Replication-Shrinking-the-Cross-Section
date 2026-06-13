"""gen_shrinkage.py — 扩展 EXT:广义收缩曲线(把收缩形状 η 当成可学旋钮)

论文把 SDF 系数先验设为 μ ~ N(0, (κ²/τ)Σ^η) 并**固定 η=2**(relative 收缩:对低方差 PC
收缩更狠)。本扩展把 η 当成可学的"收缩形状",在与四张核心图**完全一致**的 3-fold OOS-CSR²
口径下联合搜 (η, γ),检验"赋予收缩曲线更大灵活度能否超过论文固定的 η=2"。

PC 空间收缩权重(作用在 OLS 系数 μ_{P,j}/d_j 上,论文式18-24 一般化):
    w_j(η,γ) = d_j^{η-1} / (d_j^{η-1} + γ),     b_{P,j} = w_j · μ_{P,j}/d_j
  η=1 -> 常数(level,Pástor-Stambaugh);η=2 -> d_j/(d_j+γ)(KNS relative);η>2 -> 低 PC 收缩更狠。
这是一条单调(η≥1 时随 d 递增)收缩曲线;**学 η 等价于学这条曲线的形状**(与 logistic 形式
w=d^β/(d^β+c) 恒等,β=η-1)。每折 **train-only 旋转**(去 look-ahead,与 R2 同口径)。

诚实预期(印证论文):最优 η*≈2、OOS 在 η≳1.5 后基本持平、学到的曲线≈η=2 ridge,相对固定
η=2 无统计显著提升(见 R3 bootstrap CI)—— 即"线性 relative 收缩已近最优,额外灵活度无益"。
本扩展从"可解释旋钮"角度补足 fig5(自由黑箱 MLP 也未跑赢)的"自由灵活度"角度。
"""
import numpy as np
import scs_core as C
from scs_data import prepared
import scs_plot as P


def best_oos_for_eta(folds_eta, eta, d_ref, n_gamma=90, K=3):
    """对固定 η,在 **η-自适应** γ 网格上取最优 OOS,返回 (best_oos, best_gamma)。

    收缩转折点在 γ ~ d^{η-1}(权重 w=d^{η-1}/(d^{η-1}+γ) 在 γ≈d^{η-1} 处过渡),故 γ 的搜索
    范围必须随 η 缩放,否则远离 η=2 的形状会被固定网格"假性过收缩"成 ~0(网格伪影)。
    用全样本特征值 d_ref 的 d^{η-1} 跨度 ±2.5 个数量级作为每个 η 的搜索区间。
    """
    pe = np.power(np.asarray(d_ref, float), eta - 1.0)
    lo, hi = np.log10(pe.min()), np.log10(pe.max())
    gammas = np.logspace(lo - 2.5, hi + 2.5, n_gamma)
    best, bg = -np.inf, 0.0
    for g in gammas:
        oos, _ = C.eta_oos(folds_eta, g, eta=eta, K=K)
        if oos > best:
            best, bg = oos, g
    return best, bg


def compute(daily=True, n_eta=61, n_gamma=90):
    D = prepared(daily=daily)
    K = D['K']
    folds_eta = C.precompute_folds_eta(D['Rv'], K)
    d_ref = D['d']
    etas = np.linspace(0.0, 6.0, n_eta)
    best = np.empty(n_eta); best_g = np.empty(n_eta)
    for i, eta in enumerate(etas):
        best[i], best_g[i] = best_oos_for_eta(folds_eta, eta, d_ref, n_gamma, K)
    istar = int(np.argmax(best))
    eta_star, g_star, oos_star = float(etas[istar]), float(best_g[istar]), float(best[istar])
    i2 = int(np.argmin(np.abs(etas - 2.0)))
    oos_eta2, g_eta2 = float(best[i2]), float(best_g[i2])
    i1 = int(np.argmin(np.abs(etas - 1.0)))

    # 学到的收缩曲线形状(用全样本 d 画 w(d) 对照,统一各曲线到同一条 d 轴)
    d = np.sort(D['d'])[::-1]
    def wcurve(eta, g):
        de = np.power(d, eta - 1.0)
        return de / (de + g)
    return dict(etas=etas, best=best, best_g=best_g,
                eta_star=eta_star, g_star=g_star, oos_star=oos_star,
                oos_eta2=oos_eta2, g_eta2=g_eta2, d=d,
                w_eta1=wcurve(1.0, best_g[i1]), w_eta2=wcurve(2.0, g_eta2),
                w_star=wcurve(eta_star, g_star),
                folds_eta=folds_eta, D=D)


def plot(res):
    P.setup()
    import matplotlib.pyplot as plt
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 4.6))
    # (a) best OOS vs η
    axA.plot(res['etas'], res['best'], '-', color='tab:purple', lw=2.0)
    axA.axvline(2.0, color='tab:orange', ls='--', lw=1.4, label='KNS fixed $\\eta=2$')
    axA.axvline(res['eta_star'], color='tab:green', ls=':', lw=1.4,
                label=f"learned $\\eta^*$={res['eta_star']:.2f}")
    axA.axvline(1.0, color='goldenrod', ls='-.', lw=1.0, alpha=0.7, label='level $\\eta=1$')
    axA.set_xlabel('Shrinkage shape $\\eta$', fontweight='bold')
    axA.set_ylabel('Best OOS Cross-sectional $R^2$ (over $\\gamma$)', fontweight='bold')
    axA.set_title(f"(a) OOS vs shrinkage shape — $\\eta^*$={res['eta_star']:.2f} "
                  f"({res['oos_star']:.3f}) vs $\\eta$=2 ({res['oos_eta2']:.3f})", fontsize=9.5)
    axA.legend(loc='lower right', fontsize=8.5)
    axA.set_ylim(min(0.0, res['best'].max() - 0.32), res['best'].max() + 0.02)
    # (b) learned shrinkage weight curves w(d)
    ld = np.log10(res['d'])
    axB.plot(ld, res['w_eta1'], '-.', color='goldenrod', lw=1.6, label='$\\eta=1$ (level)')
    axB.plot(ld, res['w_eta2'], '--', color='tab:orange', lw=2.0, label='$\\eta=2$ (KNS)')
    axB.plot(ld, res['w_star'], '-', color='tab:green', lw=2.0,
             label=f"learned $\\eta^*$={res['eta_star']:.2f}")
    axB.set_xlabel('$\\log_{10}$ PC eigenvalue $d_j$ (high-variance $\\to$ right)', fontweight='bold')
    axB.set_ylabel('Shrinkage weight $w_j=d_j^{\\eta-1}/(d_j^{\\eta-1}+\\gamma)$', fontweight='bold')
    axB.set_title('(b) Learned shrinkage curve $\\approx$ relative ($\\eta=2$)', fontsize=9.5)
    axB.legend(loc='upper left', fontsize=8.5)
    axB.set_ylim(-0.03, 1.03)
    return P.save(fig, 'fig7_generalized_shrinkage')


if __name__ == '__main__':
    res = compute(daily=True)
    np.savez('outputs/fig7_data.npz',
             etas=res['etas'], best=res['best'], best_g=res['best_g'],
             eta_star=res['eta_star'], g_star=res['g_star'], oos_star=res['oos_star'],
             oos_eta2=res['oos_eta2'], d=res['d'],
             w_eta1=res['w_eta1'], w_eta2=res['w_eta2'], w_star=res['w_star'])
    path = plot(res)
    print(f"[EXT/fig7] 学收缩形状 η:")
    print(f"  最优 η*={res['eta_star']:.3f}  OOS={res['oos_star']:.4f}")
    print(f"  论文固定 η=2          OOS={res['oos_eta2']:.4f}")
    print(f"  增益 = {res['oos_star']-res['oos_eta2']:+.4f}  (待 R3 bootstrap 判断是否落噪声内)")
    print(f"  best OOS(η) 在 η∈[1.5,4] 的范围: "
          f"[{res['best'][(res['etas']>=1.5)].min():.4f}, {res['best'][(res['etas']>=1.5)].max():.4f}]")
    print(f"  saved: {path}")
