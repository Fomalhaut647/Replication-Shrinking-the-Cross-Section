"""scs_data.py — 统一数据加载与预处理

四张复现图与 MLP 扩展全部通过本模块取数,确保 de-market/de-vol/协方差估计/CV 划分
口径完全一致(任何口径差异都会让图与图之间不可比)。

主数据集:50 anomaly 多空组合(论文 4.2.1 节、Figure 3/4、Table 1)。
默认 daily(论文用日度收益精确估协方差),可切 monthly 做快速验证。
"""
import os
from datetime import datetime
import numpy as np
from load_managed_portfolios import load_managed_portfolios
from load_ff25 import load_ff25
import scs_core as C


def load_anomalies(daily=True):
    """加载 50 anomaly 原始多空组合收益(复用原仓库 loader,丢弃派生 instrument 列)。"""
    suffix = '_d' if daily else ''
    fn = os.path.join('Data', 'Instruments', f'managed_portfolios_anom{suffix}_50.csv')
    dates, re, mkt, anomalies = load_managed_portfolios(fn, daily, 0.2, ['rX_', 'r2_', 'r3_'])
    return dates, re, mkt, list(anomalies)


def prepared(daily=True, K=3, seed=0):
    """返回一个包含全部下游所需对象的 dict,口径统一。

    keys:
      R         : DataFrame T×N,去市场+去波动后的因子收益(保留列名)
      Rv        : R.values (ndarray)
      Sigma     : regcov(Rv) 收缩协方差(论文估 Σ)
      mu        : 样本均值向量
      T, N      : 维度
      anomalies : 因子名列表
      freq      : 年化频率(daily=252, monthly=12)
      folds     : K-fold 连续块预计算 (X_tr,y_tr,X_te,y_te)*K
      P, d, Q   : PC 空间(P=Rv·Q, d=特征值降序, Q 特征向量)
      dates     : 日期序列
    """
    np.random.seed(seed)
    dates, re, mkt, anomalies = load_anomalies(daily)
    R = C.prepare_factors(dates, re, mkt)
    Rv = R.values
    T, N = Rv.shape
    Sigma = C.regcov(Rv)
    mu = Rv.mean(0)
    folds = C.precompute_folds(Rv, K)
    P, d, Q = C.pc_rotate(Rv, Sigma)
    return dict(R=R, Rv=Rv, Sigma=Sigma, mu=mu, T=T, N=N,
                anomalies=anomalies, freq=(252 if daily else 12),
                folds=folds, P=P, d=d, Q=Q, dates=dates, K=K)


def prepared_ff25(daily=True, K=3, seed=0, t0=None, tN=None):
    """FF25 版 prepared() —— 扩展 R1。

    把经典 25 个 size/BM 组合(Ken French 公开免费数据)走与 50 anomaly **完全一致**的
    KNS 口径(de-market + de-vol + 收缩协方差 + PC 旋转 + K-fold 连续块),返回同形 dict,
    从而能直接复用 dual_penalty.scan / sparsity_frontier 跑稀疏 vs 稠密。

    样本窗默认 1963-07~2017-12,与 50 anomaly 主样本对齐以便横向比较(论文 4.1 预备分析精神)。
    返回 dict 的键与 prepared() 一致(anomalies 此处为 25 个组合标签)。
    """
    if t0 is None:
        t0 = datetime(1963, 7, 1)
    if tN is None:
        tN = datetime(2017, 12, 31)
    np.random.seed(seed)
    dates, ret, mkt, _DATA, labels = load_ff25('Data/', daily, t0, tN)
    R = C.prepare_factors(dates, ret, mkt)
    Rv = R.values
    T, N = Rv.shape
    Sigma = C.regcov(Rv)
    mu = Rv.mean(0)
    folds = C.precompute_folds(Rv, K)
    P, d, Q = C.pc_rotate(Rv, Sigma)
    return dict(R=R, Rv=Rv, Sigma=Sigma, mu=mu, T=T, N=N,
                anomalies=list(labels), freq=(252 if daily else 12),
                folds=folds, P=P, d=d, Q=Q, dates=dates, K=K)
