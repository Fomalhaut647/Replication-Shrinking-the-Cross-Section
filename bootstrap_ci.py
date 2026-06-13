"""bootstrap_ci.py — 扩展 R3:分块(block)bootstrap 置信区间(无泄漏)

给关键 OOS 数值加置信区间,用于:
  (1) fig1 头条:η=2 relative 收缩的 OOS 峰值 R²(论文核心数字)的 CI;
  (2) EXT:学到的 η*≈3.6 相对论文固定 η=2 的 OOS **增益 Δ** 是否落在噪声内(诚实性命门)。

**无泄漏分块 bootstrap**:朴素地"重采样整条序列再切 CV"会让同一原始观测同时落进 train 和
test 折 -> 泄漏 -> OOS 被抬向样本内值(实测会抬到 ~0.8,错误)。正确做法:保持真实的 K 折
连续块划分(train/test 原始时段不相交),在**每折的 train 时段、test 时段各自独立**做循环分块
重采样(块长 block_len 保序列相关),两期不相交故无泄漏。每个 bootstrap 重建该折 PC 矩并重扫 γ,
取该 η 最优 OOS;n_boot 个值取 2.5/97.5 百分位。固定 default_rng 种子,纯 CPU。
"""
import numpy as np
import scs_core as C
from scs_data import prepared


def block_resample(rows, block_len, rng):
    """在给定(连续)时段 rows 内做循环分块重采样,返回等长的原始行索引(不跨时段,无泄漏)。"""
    n = len(rows)
    nb = int(np.ceil(n / block_len))
    starts = rng.integers(0, n, size=nb)
    pos = np.concatenate([(np.arange(s, s + block_len) % n) for s in starts])[:n]
    return np.asarray(rows)[pos]


def _fe_from_rows(Rv, tr_rows, te_rows):
    """由 train/test 原始行构造一折的 PC 空间矩(train-only 旋转,与 precompute_folds_eta 同口径)。"""
    Rtr, Rte = Rv[tr_rows], Rv[te_rows]
    d, Q = np.linalg.eigh(C.regcov(Rtr))
    order = np.argsort(d)[::-1]
    d, Q = d[order], Q[:, order]
    return (d, Q.T @ Rtr.mean(0), Q.T @ C.regcov(Rte) @ Q, Q.T @ Rte.mean(0))


def _best_oos(fe, eta, d_ref, K, n_gamma=90):
    """在 η-自适应 γ 网格上取该 η 的最优 OOS(同 gen_shrinkage.best_oos_for_eta 口径)。"""
    pe = np.power(np.asarray(d_ref, float), eta - 1.0)
    gammas = np.logspace(np.log10(pe.min()) - 2.5, np.log10(pe.max()) + 2.5, n_gamma)
    best = -np.inf
    for g in gammas:
        oos, _ = C.eta_oos(fe, g, eta=eta, K=K)
        if oos > best:
            best = oos
    return best


def run(daily=True, n_boot=400, block_len=21, etas=(1.0, 2.0, 3.6), seed=0):
    D = prepared(daily=daily)
    Rv, K, d_ref = D['Rv'], D['K'], D['d']
    T = Rv.shape[0]
    etas = list(etas)
    allidx = np.arange(T)
    folds_rows = [(np.setdiff1d(allidx, te), np.asarray(te))
                  for te in C.cv_partition_contiguous(T, K)]
    rng = np.random.default_rng(seed)

    # 全样本点估计(参照):用真实 train/test,不重采样
    fe0 = [_fe_from_rows(Rv, tr, te) for (tr, te) in folds_rows]
    point = {e: _best_oos(fe0, e, d_ref, K) for e in etas}

    boot = {e: np.empty(n_boot) for e in etas}
    for b in range(n_boot):
        fe = [_fe_from_rows(Rv, block_resample(tr, block_len, rng),
                            block_resample(te, block_len, rng))
              for (tr, te) in folds_rows]
        for e in etas:
            boot[e][b] = _best_oos(fe, e, d_ref, K)

    ci = {e: (float(np.percentile(boot[e], 2.5)), float(np.percentile(boot[e], 97.5)))
          for e in etas}
    res = dict(etas=etas, n_boot=n_boot, block_len=block_len, point=point, boot=boot, ci=ci)
    if 3.6 in etas and 2.0 in etas:
        res['delta'] = boot[3.6] - boot[2.0]
    return res


if __name__ == '__main__':
    res = run(daily=True, n_boot=400, block_len=21)
    np.savez('outputs/bootstrap_ci.npz',
             etas=np.array(res['etas']),
             **{f'boot_eta{e}': res['boot'][e] for e in res['etas']},
             delta=res.get('delta', np.array([])))
    print(f"[R3] 无泄漏循环分块 bootstrap  n_boot={res['n_boot']}  block_len={res['block_len']}")
    for e in res['etas']:
        lo, hi = res['ci'][e]
        print(f"  η={e:.1f}: 点估计 OOS={res['point'][e]:.4f}  95%CI=[{lo:.4f}, {hi:.4f}]")
    if 'delta' in res:
        d = res['delta']
        dlo, dhi = np.percentile(d, 2.5), np.percentile(d, 97.5)
        print(f"  增益 Δ=OOS(η*=3.6)-OOS(η=2): 均值={d.mean():+.4f}  "
              f"95%CI=[{dlo:+.4f}, {dhi:+.4f}]  P(Δ>0)={float((d > 0).mean()):.2f}")
        print(f"  -> {'CI 含0,增益不显著(印证 η=2 近最优)' if dlo <= 0 <= dhi else 'CI 不含0,增益显著'}")
