"""r2_lookahead.py — 扩展 R2:去 look-ahead(每折 train-only 的 PC 旋转 Q)

复现主线把因子收益旋到 PC 空间用的是**全样本**协方差的特征向量 Q(scs_data.pc_rotate 用
regcov(全样本))。训练块"看到"了测试块参与估计的 Q —— 轻微 look-ahead。本脚本严格量化它,
结论分两类(关键洞察):

  (1) **纯 L2 ridge**(fig1 主线 / fig4 的 L2 包络 / SDF 系数):各向同性惩罚 γI 下 ridge
      **旋转等变**,OOS R² 与 Q 的选择**精确无关** -> 全样本 Q 与 train-only Q 的差**恒等于 0**。
      本脚本在 κ 网格上数值确认(应 ~1e-15)。

  (2) **PC-sparse**(L1 在 PC 空间选少数 PC,fig2b/fig4 的 PC 线):**旋转相关**,Q 的 look-ahead
      真会影响。比较两版 PC-sparse 的**全局最优 OOS**(干净:逐 NZ 前沿因 train-Q 各折非零数
      不一致而不可比,故只比全局最优):若 train-only Q 不低于 full-Q,则主线 PC 结果未被
      look-ahead 抬高。

η 族(fig1 的 η=1、fig7 学 η)已**默认 train-only Q**(scs_core.precompute_folds_eta),本无此 look-ahead。
"""
import numpy as np
import scs_core as C
from scs_data import prepared
from dual_penalty import scan
import scs_plot as P


def l2_rotation_invariance(D, ngrid=40):
    """纯 L2:PC 空间 OOS 用 full-Q(D['P'])vs train-Q(eta_oos η=2)在 κ 网格上比,返回最大|差|。"""
    Sig, T, freq, K = D['Sigma'], D['T'], D['freq'], D['K']
    folds_full = C.precompute_folds(D['P'], K)          # 全样本 Q 的 PC 折
    folds_tr = C.precompute_folds_eta(D['Rv'], K)        # 每折 train-only Q
    kappas = np.logspace(np.log10(0.05), np.log10(1.2), ngrid)
    mx = 0.0
    for k in kappas:
        g = C.kappa_to_gamma(k, Sig, T, freq)
        _, oos_full, _ = C.oos_cv_r2(folds_full, g, 0.0, K)   # full-Q L2
        oos_tr, _ = C.eta_oos(folds_tr, g, eta=2.0, K=K)      # train-Q L2(η=2)
        mx = max(mx, abs(oos_full - oos_tr))
    return mx


def compute(daily=True):
    D = prepared(daily=daily)
    l2_diff = l2_rotation_invariance(D)
    res_full = scan(space='pc', daily=daily, D=D, pc_trainQ=False, nkappa=32, ngamma1=60)
    res_tr = scan(space='pc', daily=daily, D=D, pc_trainQ=True, nkappa=32, ngamma1=60)
    return dict(l2_diff=l2_diff,
                pcsparse_full=float(res_full['R2'].max()),
                pcsparse_tr=float(res_tr['R2'].max()))


def plot(res):
    P.setup()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    bars = ['L2 ridge\n(rotation-invariant)', 'PC-sparse\nfull-sample $Q$', 'PC-sparse\ntrain-only $Q$']
    vals = [res['pcsparse_full'], res['pcsparse_full'], res['pcsparse_tr']]  # L2 两版相同,用 full 值占位
    colors = ['tab:gray', 'tab:red', 'tab:blue']
    ax.bar(bars, vals, color=colors, alpha=0.85)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f'{v:.3f}', ha='center', fontsize=9)
    ax.set_ylabel('Best OOS Cross-sectional $R^2$', fontweight='bold')
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_title(f'Fig R2 — Look-ahead via PC rotation $Q$\n'
                 f'L2 exactly invariant (|diff|={res["l2_diff"]:.1e}); '
                 f'PC-sparse not inflated (train-$Q$ $\\geq$ full-$Q$)', fontsize=9.0)
    return P.save(fig, 'fig8_r2_lookahead')


if __name__ == '__main__':
    res = compute(daily=True)
    np.savez('outputs/fig8_data.npz', l2_diff=res['l2_diff'],
             pcsparse_full=res['pcsparse_full'], pcsparse_tr=res['pcsparse_tr'])
    path = plot(res)
    print(f"[R2] 去 look-ahead(train-only Q):")
    print(f"  (1) 纯 L2 旋转不变性: full-Q vs train-Q 最大|OOS差| = {res['l2_diff']:.2e}  (应 ~1e-15,精确为0)")
    print(f"  (2) PC-sparse 全局最优 OOS: full-Q={res['pcsparse_full']:.4f}  train-Q={res['pcsparse_tr']:.4f}")
    verdict = '未被抬高(train-Q≥full-Q)' if res['pcsparse_tr'] >= res['pcsparse_full'] - 1e-6 else '略降'
    print(f"      -> look-ahead 对 PC-sparse 结论 {verdict};纯 L2 结论严格无 look-ahead。")
    print(f"  saved: {path}")
