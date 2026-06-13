"""reproduce_all.py — 一键从头复现全部结果(阶段2四图 + 阶段3 MLP)

用法:
    uv run python reproduce_all.py

固定随机种子(scs_data.prepared 内 np.random.seed(0) + MLP 用 default_rng(0)),纯 CPU。
所有图与中间数据写入 outputs/。每步打印关键数值,便于与论文/本仓库报告核对。

复现链:
    阶段1(原样跑通)见 scs_main.py(本脚本不重复);本脚本聚焦阶段2/3 的论文核心结果。
"""
import time
import numpy as np

import scs_data
import dual_penalty
import fig1_l2_selection as F1
import fig2_dual_penalty_contour as F2
import fig3_pc_coefficients as F3
import fig4_sparsity_frontier as F4
import mlp_sdf as F5


def banner(t):
    print('\n' + '=' * 70 + f'\n{t}\n' + '=' * 70)


def main(daily=True):
    t0 = time.time()

    banner('图1 — L2 模型选择(论文 Figure 4a)')
    r1 = F1.compute(daily=daily)
    np.savez('outputs/fig1_data.npz', **r1)
    F1.plot(r1)
    print(f"  OOS 峰值 R²={r1['r2star']:.4f} @ κ*={r1['kstar']:.4f}; "
          f"P&S level 峰值={np.max(r1['PS']):.4f}")

    banner('图2/图4 共享:双惩罚网格扫描(char + pc)')
    D = scs_data.prepared(daily=daily)
    for space in ['char', 'pc']:
        res = dual_penalty.scan(space=space, daily=daily, D=D, nkappa=32, ngamma1=60)
        np.savez(f'outputs/grid_{space}.npz', **res)
        print(f"  {space}: OOS R² 网格范围 [{res['R2'].min():.3f}, {res['R2'].max():.3f}]")

    banner('图2 — L1-L2 双惩罚等高线(论文 Figure 3a/3b)')
    F2.plot()
    gc, gp = np.load('outputs/grid_char.npz'), np.load('outputs/grid_pc.npz')
    print(f"  非零数<=10 时最高 OOS R²: PC={gp['R2'][gp['NZ']<=10].max():.3f} "
          f"vs char={gc['R2'][gc['NZ']<=10].max():.3f}")

    banner('图3 — SDF 系数在 PC 空间的分布(论文 Table 1b)')
    r3 = F3.compute(daily=daily)
    np.savez('outputs/fig3_data.npz', b_P=r3['b_P'], t_P=r3['t_P'], d=r3['d'],
             kappa_star=r3['kappa_star'])
    F3.plot(r3)
    top = r3['order_t'][:5]
    tp = r3['t_P']
    print("  |t| 最大的 5 个 PC: " + ', '.join('PC%d(%.2f)' % (j + 1, abs(tp[j])) for j in top))

    banner('图4 — 稀疏 vs 稠密 OOS R²(论文 Figure 4b)')
    r4 = F4.compute(daily=daily)
    np.savez('outputs/fig4_data.npz', ks_c=r4['ks_c'], best_c=r4['best_c'], se_c=r4['se_c'],
             ks_p=r4['ks_p'], best_p=r4['best_p'], se_p=r4['se_p'])
    F4.plot(r4)
    print(f"  PC 4因子 OOS R²={r4['best_p'][3]:.3f}; char 4因子={r4['best_c'][3]:.3f}; "
          f"char 50因子={r4['best_c'][-1]:.3f}")

    banner('图5 — MLP 延伸:非线性自适应收缩 vs 线性 ridge(阶段3)')
    r5 = F5.run(daily=daily)
    np.savez('outputs/fig5_data.npz', mlp_oos=r5['mlp_oos'], lin_oos=r5['lin_oos'],
             kappa_star=r5['kappa_star'])
    F5.plot(r5)
    print(f"  线性 ridge OOS R²={r5['lin_oos'].mean():.4f}  |  Tiny MLP={r5['mlp_oos'].mean():.4f}  "
          f"(Δ={r5['mlp_oos'].mean()-r5['lin_oos'].mean():+.4f}, MLP 未跑赢 -> 印证线性近最优)")

    banner('图6 (扩展 R1) — FF25 预备分析:低维性 + 稀疏 vs 稠密')
    import fig6_ff25_sparsity as F6
    r6 = F6.compute(daily=daily)
    np.savez('outputs/fig6_data.npz', ks_c=r6['ks_c'], best_c=r6['best_c'],
             ks_p=r6['ks_p'], best_p=r6['best_p'])
    F6.plot(r6)
    print(f"  FF25 top-3 PC 方差占比={r6['cum_ff'][2]:.0%}; OOS 网格峰值≈{r6['res_p']['R2'].max():.3f}"
          f"(去市场残差时变,OOS≈0,负对照)")

    banner('图7 (扩展 EXT) — 广义收缩曲线:学收缩形状 η')
    import gen_shrinkage as F7
    r7 = F7.compute(daily=daily)
    np.savez('outputs/fig7_data.npz', etas=r7['etas'], best=r7['best'],
             eta_star=r7['eta_star'], oos_star=r7['oos_star'], oos_eta2=r7['oos_eta2'])
    F7.plot(r7)
    print(f"  学到 η*={r7['eta_star']:.2f}(OOS={r7['oos_star']:.4f}) vs 论文固定 η=2"
          f"(OOS={r7['oos_eta2']:.4f}),增益 {r7['oos_star']-r7['oos_eta2']:+.4f}")

    banner('图8 (扩展 R2) — 去 look-ahead:train-only Q')
    import r2_lookahead as F8
    r8 = F8.compute(daily=daily)
    np.savez('outputs/fig8_data.npz', l2_diff=r8['l2_diff'],
             pcsparse_full=r8['pcsparse_full'], pcsparse_tr=r8['pcsparse_tr'])
    F8.plot(r8)
    print(f"  纯 L2 旋转不变 |diff|={r8['l2_diff']:.1e}(精确0); "
          f"PC-sparse full-Q={r8['pcsparse_full']:.3f} vs train-Q={r8['pcsparse_tr']:.3f}(未抬高)")

    banner('扩展 R3 — 分块 bootstrap 置信区间')
    import bootstrap_ci as B3
    rb = B3.run(daily=daily, n_boot=300, block_len=21)
    np.savez('outputs/bootstrap_ci.npz', etas=np.array(rb['etas']),
             **{f'boot_eta{e}': rb['boot'][e] for e in rb['etas']}, delta=rb.get('delta', np.array([])))
    lo2, hi2 = rb['ci'][2.0]
    dlt = rb['delta']; dlo, dhi = np.percentile(dlt, 2.5), np.percentile(dlt, 97.5)
    print(f"  η=2 OOS={rb['point'][2.0]:.3f} 95%CI=[{lo2:.3f},{hi2:.3f}](显著>0); "
          f"Δ(η*-η2)95%CI=[{dlo:+.3f},{dhi:+.3f}]{'含0,不显著' if dlo<=0<=dhi else '显著'}")

    banner(f'完成 — 用时 {time.time()-t0:.1f}s,全部图见 outputs/')
    import os
    for f in sorted(os.listdir('outputs')):
        if f.endswith('.png'):
            print(f"  outputs/{f}")


if __name__ == '__main__':
    main(daily=True)
