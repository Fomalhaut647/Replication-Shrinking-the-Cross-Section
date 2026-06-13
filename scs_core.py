"""scs_core.py — Kozak, Nagel & Santosh (2020) 方法的统一、干净实现(阶段2/3复现)

本模块在**不改动原仓库逻辑**的前提下,把论文方法抽成可复用的纯函数,供阶段2(四个
核心图)与阶段3(MLP 延伸)调用。核心原则:**所有 OOS 横截面 R² 严格采用论文式(30)的
口径**,与原仓库 `cross_validate.py` 的 `bootstrp_obj_CSR2` 完全一致,从而保证四个图
(及 MLP 对比)在同一把尺子下度量。

复用原仓库:
  - utils.demarket : 去市场 beta(无条件)
  - utils.regcov   : 带 flat-Wishart 先验的收缩协方差(论文用它估 Σ)
  - utils.l2est    : 纯 L2 ridge 解(交叉验证一致性的金标准)

新增:
  - enet_coef      : 双惩罚 elastic net 坐标下降(论文式28)
  - pc_rotate      : 旋转到主成分(PC)空间(论文式10-11)
  - oos_cv_r2      : 高效统一 K-fold 连续块 OOS-CV(复刻式30口径)

论文关键公式索引:
  (22) 纯L2 :  b = (Σ + γI)^{-1} μ,         γ = freq·tr(Σ)/(T·κ²)
  (28) 双惩罚: b = argmin (μ-Σb)'Σ⁻¹(μ-Σb) + γ₂ b'b + γ₁‖b‖₁
  (30) OOS R²: 1 - (μ₂-Σ₂b)'(μ₂-Σ₂b) / (μ₂'μ₂)
"""
import numpy as np
import pandas as pd
from utils import demarket, regcov, l2est


# ============================================================================
# 1. 数据准备:de-market + de-vol(复刻 SCS_L2est 的无条件预处理)
# ============================================================================
def prepare_factors(dates, re, mkt):
    """对原始多空组合收益做与原仓库一致的预处理。

    步骤(对齐 SCS_L2est.py 的 demarket_unconditionally + devol_unconditionally):
      1. 去市场:对每列回归剔除市场 beta(全样本估计 beta)。
      2. 去波动:把每列标准化到市场超额收益的波动率(保持杠杆可比)。

    返回: r0 (DataFrame, T×N) 处理后的因子收益,保留列名(anomaly 标签)。
    """
    re = re.copy()
    mkt = mkt.copy()
    re.index = dates.values
    mkt.index = dates.values
    r0, _ = demarket(re, mkt)                                   # 去市场 beta
    r0 = r0.divide(r0.std(axis=0), axis=1).multiply(mkt.std())  # de-vol 到市场波动
    return r0


# ============================================================================
# 2. κ <-> γ 映射(论文式29 的经济化参数;freq 因子使 κ 为年化 root-E[SR²])
# ============================================================================
def kappa_to_gamma(kappa, Sigma, T, freq):
    """κ(root expected max squared Sharpe) -> L2 惩罚强度 γ。与 kappa2pen 一致。"""
    return freq * np.trace(Sigma) / T / (kappa ** 2)


# ============================================================================
# 3. 估计器
# ============================================================================
def l2_coef(Sigma, mu, gamma):
    """纯 L2 ridge 解,论文式(22): b = (Σ + γI)^{-1} μ。"""
    n = Sigma.shape[0]
    return np.linalg.solve(Sigma + gamma * np.eye(n), np.asarray(mu, float).ravel())


def enet_coef(Sigma, mu, gamma2, gamma1, b0=None, max_iter=5000, tol=1e-10):
    """双惩罚 elastic net,论文式(28),坐标下降求解。

    目标(展开 (μ-Σb)'Σ⁻¹(μ-Σb) = const - 2b'μ + b'Σb):
        f(b) = b'(Σ+γ₂I)b - 2 b'μ + γ₁‖b‖₁
    令 A = Σ + γ₂I。对坐标 j 的一阶条件(次梯度=0):
        A_jj b_j = μ_j - Σ_{k≠j} A_jk b_k - (γ₁/2) sign(b_j)
    即软阈值更新:
        b_j = soft(μ_j - Σ_{k≠j} A_jk b_k, γ₁/2) / A_jj,  soft(z,t)=sign(z)·max(|z|-t,0)

    维护 Ab=A·b 做增量更新,O(n) 每坐标。γ₁=0 时退化为纯 L2(== l2_coef),已在自测验证。
    """
    n = Sigma.shape[0]
    A = Sigma + gamma2 * np.eye(n)
    mu = np.asarray(mu, float).ravel()
    b = np.zeros(n) if b0 is None else np.asarray(b0, float).copy()
    dA = np.diag(A).copy()
    Ab = A @ b
    thr = gamma1 / 2.0
    for _ in range(max_iter):
        dmax = 0.0
        for j in range(n):
            rho = mu[j] - (Ab[j] - dA[j] * b[j])      # μ_j - Σ_{k≠j} A_jk b_k
            bj = np.sign(rho) * max(abs(rho) - thr, 0.0) / dA[j]
            db = bj - b[j]
            if db != 0.0:
                Ab += A[:, j] * db                    # 增量更新 Ab
                b[j] = bj
                ad = abs(db)
                if ad > dmax:
                    dmax = ad
        if dmax < tol:
            break
    return b


# ============================================================================
# 4. PC 旋转(论文式10-11: Σ=QDQ', P_t = Q'F_t)
# ============================================================================
def pc_rotate(R, Sigma=None):
    """把因子收益矩阵 R(T×N) 旋转到 PC 空间。

    返回 (P, d, Q):
      P : T×N 的 PC 组合收益 = R·Q
      d : 长度 N 的特征值(降序,= PC 方差)
      Q : N×N 特征向量矩阵(列为特征向量)
    用收缩协方差 regcov(R) 做特征分解,与估计 Σ 保持一致。
    """
    if Sigma is None:
        Sigma = regcov(R)
    d, Q = np.linalg.eigh(Sigma)        # eigh:对称矩阵,升序
    order = np.argsort(d)[::-1]         # 转降序(高方差 PC 在前)
    d = d[order]
    Q = Q[:, order]
    P = np.asarray(R) @ Q
    return P, d, Q


# ============================================================================
# 5. 统一 OOS 交叉验证(连续块 K-fold,复刻式30口径)
# ============================================================================
def cv_partition_contiguous(T, K):
    """连续块划分(非随机,按时间顺序),复刻 cross_validate.cvpartition_contiguous。"""
    s = T // K
    idx = [list(range(s * i, s * (i + 1))) for i in range(K - 1)]
    idx.append(list(range(s * (K - 1), T)))
    return idx


def precompute_folds(R, K):
    """预计算每个 fold 的训练/测试矩(regcov + mean),供网格扫描复用。

    R: T×N ndarray。返回 list[(X_tr, y_tr, X_te, y_te)]。
    口径严格对齐 cross_validate.bootstrp_handler:训练=K-1块,测试=留出块,
    协方差用 regcov(收缩),均值用样本均值。
    """
    R = np.asarray(R, float)
    T = R.shape[0]
    folds = cv_partition_contiguous(T, K)
    allidx = np.arange(T)
    out = []
    for te in folds:
        tr = np.setdiff1d(allidx, te)
        Rtr, Rte = R[tr], R[te]
        out.append((regcov(Rtr), Rtr.mean(0), regcov(Rte), Rte.mean(0)))
    return out


def csr2(y_hat, y):
    """横截面 R²,论文式(30)的核= 1 - (ŷ-y)'(ŷ-y)/(y'y)。"""
    y = np.asarray(y, float).ravel()
    d = np.asarray(y_hat, float).ravel() - y
    return 1.0 - d.dot(d) / y.dot(y)


def enet_lambda1_max(mu):
    """使 elastic net 全部系数归零的最小 γ1。

    坐标下降从 b=0 起点时 ρ_j=μ_j,软阈值 γ1/2 ≥ max|μ_j| 即令所有 b_j=0。
    故 γ1_max = 2·max|μ_j|。图2/图4 的 γ1 网格以此为上界自适应展开。
    """
    return 2.0 * np.max(np.abs(np.asarray(mu, float)))


def enet_path(Sigma, mu, gamma2, gamma1_grid, warm=True):
    """沿 γ1 网格(建议**降序**:稀疏->稠密)warm-start 求解 elastic net。

    返回 B (len(gamma1_grid) × N)。warm-start 用上一个(更稀疏)解做初值,大幅减少
    坐标下降迭代次数;γ1 降序时解逐渐稠密,warm-start 收敛快且稳定。
    """
    n = Sigma.shape[0]
    B = np.empty((len(gamma1_grid), n))
    b0 = None
    for i, g1 in enumerate(gamma1_grid):
        b = enet_coef(Sigma, mu, gamma2, g1, b0=(b0 if warm else None))
        B[i] = b
        b0 = b
    return B


def pc_space_l2(P, d, gamma, T):
    """在 PC 空间直接估纯 L2 的 SDF 系数与标准误(论文式24 + 式23)。

    P: T×N PC 收益; d: 特征值(=PC 方差); gamma: L2 惩罚; T: 样本量。
      b_P,j = μ_P,j / (d_j + γ)            —— 式(24) 的等价形式(Σ_P=diag(d))
      var(b_P) = (1/T)·diag( (diag(d)+γI)^{-1} )  —— 式(23) 在 PC 空间
    返回 (b_P, se_P, mu_P)。
    """
    mu_P = np.asarray(P).mean(0)
    denom = d + gamma
    b_P = mu_P / denom
    se_P = np.sqrt((1.0 / T) * (1.0 / denom))
    return b_P, se_P, mu_P


def precompute_folds_eta(Rv, K):
    """为一般-η 收缩族预计算每折的 PC 空间矩(每折 **只用训练块** 估特征向量 Q_tr)。

    这是 R2/R4/EXT 共用的基座。对每个 fold:
      Q_tr, d_tr = eigh(regcov(Rtr))           —— 训练块特征向量/值(降序),去 look-ahead
      mu_P_tr    = Q_tr' · mean(Rtr)            —— 训练均值投到训练-PC 坐标
      Sig_te_pc  = Q_tr' · regcov(Rte) · Q_tr   —— 测试协方差转到同一训练-PC 坐标
      mu_te_pc   = Q_tr' · mean(Rte)
    返回 list[(d_tr, mu_P_tr, Sig_te_pc, mu_te_pc)]。eigh 每折只算一次,供 κ×η 网格复用。

    关键不变量:对 η=2,本预计算 + eta_oos 与 char 空间的 oos_cv_r2 **精确**相等(旋转等变;
    见 _selftest_core 验证),故可安全用作 ridge 基线;对 η≠2 才显出旋转依赖,此时 train-only Q
    是去 look-ahead 的正确口径。
    """
    Rv = np.asarray(Rv, float)
    T = Rv.shape[0]
    out = []
    for te in cv_partition_contiguous(T, K):
        tr = np.setdiff1d(np.arange(T), te)
        Rtr, Rte = Rv[tr], Rv[te]
        d, Q = np.linalg.eigh(regcov(Rtr))
        order = np.argsort(d)[::-1]
        d, Q = d[order], Q[:, order]
        out.append((d, Q.T @ Rtr.mean(0), Q.T @ regcov(Rte) @ Q, Q.T @ Rte.mean(0)))
    return out


def eta_oos(folds_eta, gamma2, eta=2.0, K=3, kfold_adjust=True, weights=None):
    """一般-η 收缩族的 OOS-CV,返回 (OOS_R2, se_OOS)。

    PC 空间收缩(论文式24 的一般化,见式18-22 推导):
        b_P,j = μ_P,j / (d_j + γ · d_j^{2-η})
              = w_j · (μ_P,j / d_j),   w_j = d_j^{η-1}/(d_j^{η-1}+γ)   作用在 OLS 系数上的收缩权重
      η=2 -> 分母 d_j+γ,relative 收缩(论文基线 ridge);
      η=1 -> 分母 d_j(1+γ),level 收缩(字面 Pástor-Stambaugh,所有 PC 等比例);
      η=0 -> 对低方差 PC 几乎不收缩(近似套利,论文论证不合理)。
    可选 weights:长度 N 的自定义收缩权重 w_j∈[0,1](EXT 的单调收缩曲线用),给定时
      b_P,j = weights_j · (μ_P,j / d_j),覆盖 η 公式。
    γ 随折按 1/(1-1/K) 放大以对齐 κ 口径(与 oos_cv_r2 一致)。
    """
    f = 1.0 / (1.0 - 1.0 / K) if kfold_adjust else 1.0
    g = gamma2 * f
    oos = []
    for (d, mu_P, Sig_te_pc, mu_te_pc) in folds_eta:
        if weights is None:
            b_P = mu_P / (d + g * np.power(d, 2.0 - eta))
        else:
            b_P = np.asarray(weights, float) * (mu_P / d)
        oos.append(csr2(Sig_te_pc @ b_P, mu_te_pc))
    oos = np.array(oos)
    return float(np.mean(oos)), float(np.std(oos) / np.sqrt(len(oos)))


def oos_cv_r2(folds_data, gamma2, gamma1=0.0, K=3, kfold_adjust=True, warm=None):
    """在预计算的 folds 上做 OOS-CV,返回 (IS_R2, OOS_R2, se_OOS)。

    估计器:γ₁>0 用 enet_coef,γ₁=0 用 l2_coef(更快更稳)。
    kfold_adjust:复刻原仓库 lCV=l/(1-1/K) —— CV 训练集只有 (1-1/K)·T 行,为保持与
      全样本同一个 κ(γ∝1/T),把惩罚放大 1/(1-1/K)。
    返回的 OOS R² = K 个 fold 上式(30)的均值;se = fold 间标准差/√K。
    """
    f = 1.0 / (1.0 - 1.0 / K) if kfold_adjust else 1.0
    g2 = gamma2 * f      # 仅 L2 惩罚随折大小调整(γ2∝1/T,见原仓库 lCV)
    g1 = gamma1          # L1 惩罚在 HJ 目标中无 1/T 依赖,不缩放(保持稀疏度口径一致)
    is_list, oos_list = [], []
    for (Xtr, ytr, Xte, yte) in folds_data:
        if g1 > 0:
            b = enet_coef(Xtr, ytr, g2, g1, b0=warm)
        else:
            b = l2_coef(Xtr, ytr, g2)
        is_list.append(csr2(Xtr @ b, ytr))
        oos_list.append(csr2(Xte @ b, yte))
    oos = np.array(oos_list)
    return float(np.mean(is_list)), float(np.mean(oos)), float(np.std(oos) / np.sqrt(len(oos)))
