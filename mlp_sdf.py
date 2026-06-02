"""mlp_sdf.py — 阶段3:用极小 MLP 学习"非线性自适应收缩",对比线性 ridge

论文线性估计在 PC 空间等价于对每个 PC 施加 ridge 收缩:
    b_P,j = μ_P,j/(d_j+γ),  γ 为全局常数(由 κ-CV 选)。
本扩展让收缩强度随 PC 方差**非线性自适应**:
    γ_j = γ_base(κ*) · exp( SCALE·tanh( MLP(log d_j) ) ),   b_j = μ_j/(d_j+γ_j)
其中 γ_base 是线性 ridge 在最优 κ* 下的全局惩罚。MLP 输出经 tanh 限幅,使 γ_j 被**锚定**在
ridge 的 [e^-SCALE, e^SCALE] 倍区间内:
  - 当 MLP 输出恒 0 -> γ_j≡γ_base,**精确退化为论文 ridge**(故必不劣化太多、数值必稳定);
  - MLP 可学"哪些方差档位需要相对更强/更弱收缩"的非线性形状。

为何这样设计(经多轮诊断得到的关键教训):
  SDF 系数 b∝μ 被噪声主导,自由 MLP 会系统性**欠收缩**小方差 PC 导致 OOS 崩溃
  (b=w·μ/d 参数化对小 d 病态)。把 γ 锚定在 ridge 附近,既保证稳定,又能公平检验
  "非线性收缩能否超过常数收缩"。

诚实预期:论文主张定价信息分散、线性 relative-shrinkage 已近最优。若 MLP 学到的 γ(d)≈常数、
OOS R² 不显著超过线性,正是对论文的有力佐证(绝不调参调成"赢")。

口径与防泄漏:
  - **系数估计与评估对 test 块 fold-honest**:MLP 仅在 train 块内部 3 折训练,系数 b 与标准化只用 train 块,
    test 块仅进入最终打分 CSR2(Σ_test·b, μ_test);MLP 与 linear ridge 用同一 κ*、同一口径,唯一差别是 γ 的产生方式。
  - **PC 旋转矩阵 Q 用全样本估计**(同图2/4),属轻微 look-ahead;但日度 T 大使 Q 极稳定(train-only vs 全样本
    OOS 差 <0.0005),且对 MLP 与 linear **同等影响**、不偏倚二者对比,与论文全样本正交化做法一致。
  - 锚定后 MLP 学到的是 over/under 混合的 twist(欠收缩最大 PC、过收缩中高方差 PC),净效果不改善 OOS。
"""
import numpy as np
from scipy.optimize import minimize
import scs_core as C
from scs_data import prepared
import scs_plot as P

H = 16          # 隐层神经元数(极小)
NIN = 1         # 输入:仅 log d(纯方差->收缩映射;加 in-sample 均值会过拟合噪声)
SCALE = 0.5     # tanh 限幅:γ 锚定在 ridge 的 [e^-0.5, e^0.5]≈[0.61, 1.65] 倍内(合理先验)
REG = 3e-3      # MLP 权重 L2 正则
SEED = 0


# ---------------- 参数打平/还原 ----------------
def _shapes():
    return [(H, NIN), (H,), (1, H), (1,)]

def _unpack(theta):
    out, i = [], 0
    for sh in _shapes():
        n = int(np.prod(sh)); out.append(theta[i:i+n].reshape(sh)); i += n
    return out

def _pack(W1, b1, W2, b2):
    return np.concatenate([W1.ravel(), b1.ravel(), W2.ravel(), b2.ravel()])

def _init(rng):
    W1 = rng.standard_normal((H, NIN)) * 0.3
    b1 = np.zeros(H)
    W2 = rng.standard_normal((1, H)) * 0.3
    b2 = np.zeros(1)                 # 初始 MLP≈0 -> γ≈γ_base -> 退化 ridge
    return _pack(W1, b1, W2, b2)


# ---------------- 前向:输出限幅 log-收缩-乘子 s∈[-SCALE,SCALE] ----------------
def forward(theta, X):
    W1, b1, W2, b2 = _unpack(theta)
    z1 = X @ W1.T + b1
    h = np.tanh(z1)
    z2 = h @ W2.T + b2               # (N,1)
    t = np.tanh(z2)
    s = (SCALE * t).ravel()          # γ = γ_base·exp(s)
    return s, (X, h, t, W1, W2)


def _features(Pblock):
    """返回 [log d](标准化前) 及 (d, μ)。仅用 PC 方差作输入。"""
    d = np.maximum(Pblock.var(0, ddof=0), 1e-12)
    mu = Pblock.mean(0)
    return np.log(d).reshape(-1, 1), d, mu


def _standardize(feat, scaler=None):
    if scaler is None:
        scaler = (feat.mean(0), feat.std(0) + 1e-12)
    m, sd = scaler
    return (feat - m) / sd, scaler


# ---------------- 单折 loss + 解析梯度(口径同评估;b 有界且锚定 ridge) ----------------
def _single_loss_grad(theta, Xnn, Xval, d_fit, mu_fit, mu_val, gbase):
    s, cache = forward(theta, Xnn)
    X_, h, t, W1, W2 = cache
    gamma = gbase * np.exp(s)
    denom = d_fit + gamma
    b = mu_fit / denom
    yhat = Xval @ b
    TSS = float(mu_val @ mu_val)
    diff = yhat - mu_val
    loss = float(diff @ diff) / TSS - 1.0          # = -CSR2
    dyhat = (2.0 / TSS) * diff
    dL_db = Xval @ dyhat
    # db/ds = (-μ/denom²)·(dγ/ds), dγ/ds = γ;  s = SCALE·tanh(z2) -> ds/dz2 = SCALE·(1-t²)
    g_s = dL_db * (-mu_fit / denom ** 2) * gamma
    g_z2 = (g_s * (SCALE * (1.0 - t.ravel() ** 2))).reshape(-1, 1)
    dW2 = g_z2.T @ h
    db2 = g_z2.sum(0)
    g_h = g_z2 @ W2
    g_z1 = g_h * (1.0 - h ** 2)
    dW1 = g_z1.T @ X_
    db1 = g_z1.sum(0)
    return loss, _pack(dW1, db1, dW2, db2)


def loss_grad(theta, batches):
    """跨内部折平均的 loss 与梯度 + 权重 L2 正则。batches:(Xnn,Xval,d_fit,μ_fit,μ_val,gbase)。"""
    tot_l = 0.0
    tot_g = np.zeros_like(theta)
    for (Xnn, Xval, d_f, mf, mv, gb) in batches:
        l, g = _single_loss_grad(theta, Xnn, Xval, d_f, mf, mv, gb)
        tot_l += l; tot_g += g
    n = len(batches)
    tot_l /= n; tot_g /= n
    W1, b1, W2, b2 = _unpack(theta)
    tot_l += REG * (float((W1 ** 2).sum()) + float((W2 ** 2).sum()))
    gW1, gb1, gW2, gb2 = _unpack(tot_g)
    gW1 += 2 * REG * W1; gW2 += 2 * REG * W2
    return tot_l, _pack(gW1, gb1, gW2, gb2)


# ---------------- 训练 / 预测 ----------------
def train_on_block(P_train, kappa_star, freq, rng, n_inner=3, restarts=3):
    T = P_train.shape[0]
    parts = C.cv_partition_contiguous(T, n_inner)
    feat_all, _, _ = _features(P_train)
    _, scaler = _standardize(feat_all)
    allidx = np.arange(T)
    batches = []
    for val in parts:
        fit = np.setdiff1d(allidx, val)
        Pf = P_train[fit]
        feat_f, d_f, mu_f = _features(Pf)
        Xnn, _ = _standardize(feat_f, scaler)
        gbase = C.kappa_to_gamma(kappa_star, C.regcov(Pf), len(fit), freq)
        Xval = C.regcov(P_train[val]); mu_v = P_train[val].mean(0)
        batches.append((Xnn, Xval, d_f, mu_f, mu_v, gbase))
    best_theta, best_loss = None, np.inf
    for _ in range(restarts):
        res = minimize(loss_grad, _init(rng), args=(batches,), jac=True,
                       method='L-BFGS-B', options={'maxiter': 800, 'ftol': 1e-11})
        if res.fun < best_loss:
            best_loss, best_theta = res.fun, res.x
    return best_theta, scaler


def predict_b(theta, scaler, P_block, gbase):
    feat, d, mu = _features(P_block)
    Xnn, _ = _standardize(feat, scaler)
    s, _ = forward(theta, Xnn)
    gamma = gbase * np.exp(s)
    b = mu / (d + gamma)
    w = d / (d + gamma)
    return b, w, d, gamma


# ---------------- outer K-fold 对比 ----------------
def run(daily=True, kappa_star=None):
    D = prepared(daily=daily)
    Pm, T, freq, N, K = D['P'], D['T'], D['freq'], D['N'], D['K']
    if kappa_star is None:
        try:
            kappa_star = float(np.load('outputs/fig1_data.npz')['kstar'])
        except Exception:
            kappa_star = 0.28
    folds_idx = C.cv_partition_contiguous(T, K)
    rng = np.random.default_rng(SEED)
    mlp_oos, lin_oos, wcurves = [], [], []
    for te in folds_idx:
        tr = np.setdiff1d(np.arange(T), te)
        P_tr, P_te = Pm[tr], Pm[te]
        Xte = C.regcov(P_te); yte = P_te.mean(0)
        d_tr = np.maximum(P_tr.var(0, ddof=0), 1e-12); mu_tr = P_tr.mean(0)
        gbase = C.kappa_to_gamma(kappa_star, C.regcov(P_tr), len(tr), freq)
        # 线性 ridge(同 κ*、同口径)
        b_lin = mu_tr / (d_tr + gbase)
        lin_oos.append(C.csr2(Xte @ b_lin, yte))
        w_lin = d_tr / (d_tr + gbase)
        # MLP 自适应收缩
        theta, scaler = train_on_block(P_tr, kappa_star, freq, rng)
        b_mlp, w_mlp, d_used, gamma_mlp = predict_b(theta, scaler, P_tr, gbase)
        mlp_oos.append(C.csr2(Xte @ b_mlp, yte))
        wcurves.append((d_used, w_mlp, w_lin))
    return dict(mlp_oos=np.array(mlp_oos), lin_oos=np.array(lin_oos),
                kappa_star=kappa_star, wcurves=wcurves, K=K)


def plot(res):
    P.setup()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    # (a) 学到的收缩曲线
    ax = axes[0]
    d, w_mlp, w_lin = res['wcurves'][0]
    o = np.argsort(d)
    ax.plot(d[o], w_lin[o], '-', color='tab:blue', lw=2.0, label='Linear ridge  $w=d/(d+\\gamma)$')
    ax.scatter(d, w_mlp, s=22, color='tab:red', alpha=0.8, label='MLP adaptive  $w=d/(d+\\gamma(d))$')
    ax.set_xscale('log')
    ax.set_xlabel('PC variance (eigenvalue) $d_j$', fontweight='bold')
    ax.set_ylabel('Shrinkage weight $w_j$', fontweight='bold')
    ax.set_title('(a) Learned shrinkage function: MLP vs ridge', fontsize=10)
    ax.legend(loc='upper left')
    # (b) OOS R² 对比
    ax = axes[1]
    m_mean, m_se = res['mlp_oos'].mean(), res['mlp_oos'].std() / np.sqrt(res['K'])
    l_mean, l_se = res['lin_oos'].mean(), res['lin_oos'].std() / np.sqrt(res['K'])
    bars = ax.bar(['Linear ridge\n(paper)', 'Tiny MLP\n(extension)'], [l_mean, m_mean],
                  yerr=[l_se, m_se], capsize=6, color=['tab:blue', 'tab:red'], alpha=0.85, width=0.55)
    for b, v in zip(bars, [l_mean, m_mean]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f'{v:.3f}', ha='center', fontweight='bold')
    ax.set_ylabel('OOS Cross-sectional $R^2$ (3-fold)', fontweight='bold')
    ax.set_ylim(0, max(0.32, 1.25 * max(m_mean, l_mean)))
    ax.set_title('(b) Same OOS yardstick: MLP vs linear', fontsize=10)
    fig.suptitle('Fig 5 — ML extension: tiny MLP adaptive shrinkage vs linear ridge (50 anomaly PCs)',
                 fontsize=11.5, fontweight='bold')
    fig.tight_layout()
    return P.save(fig, 'fig5_mlp_extension')


if __name__ == '__main__':
    res = run(daily=True)
    np.savez('outputs/fig5_data.npz', mlp_oos=res['mlp_oos'], lin_oos=res['lin_oos'],
             kappa_star=res['kappa_star'])
    path = plot(res)
    m, l = res['mlp_oos'].mean(), res['lin_oos'].mean()
    print(f"[阶段3 MLP] κ*={res['kappa_star']:.4f}")
    print(f"  线性 ridge OOS R² (per fold): {np.round(res['lin_oos'],4)}  -> mean={l:.4f}")
    print(f"  Tiny MLP   OOS R² (per fold): {np.round(res['mlp_oos'],4)}  -> mean={m:.4f}")
    verdict = "MLP 略胜" if m > l + 0.005 else ("基本持平" if abs(m - l) <= 0.005 else "MLP 未胜出")
    print(f"  结论: {verdict} (Δ={m-l:+.4f})")
    print(f"  saved: {path}")
