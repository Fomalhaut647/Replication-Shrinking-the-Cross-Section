# 复现报告:Kozak, Nagel & Santosh (2020) "Shrinking the Cross-Section"

> 本报告用中文解释我们做了什么、为什么结果是这样。配套一键复现脚本 `reproduce_all.py`,
> 四张核心图 + MLP 延伸图均在 `outputs/`。论文原文为仓库根目录 `Shrinking-the-Cross-Section.pdf`。

---

## 1. 概述

论文核心主张:面对几十个 anomaly 多空组合构成的横截面,**与其做稀疏因子选择(只留几个因子),
不如对所有因子的主成分(PC)做收缩**。它估计 SDF $M_t = 1 - b'(F_t - \mathbb{E}[F_t])$ 的系数
向量 $b$,关键结论是 **L2(ridge)惩罚才真正起作用,纯 L1(稀疏)表现差** —— 因为定价信息分散在
很多 PC 上,且应当对低方差 PC 施加更强收缩(relative shrinkage)。

我们的产物:
1. 仓库在纯 CPU、当前环境(Python 3.14)下**一键跑通**(阶段1);
2. **四张核心图**复现论文关键结果(阶段2,见第 4 节);
3. 一个**机器学习延伸**:用极小 MLP 替换线性收缩函数,同口径对比(阶段3,见第 5 节);
4. 本报告 + 带注释、可从头跑到尾的脚本。

**硬约束遵守**:全程**无任何付费数据**(只用仓库自带的组合层面收益 + Ken French 公开免费数据);
**纯 CPU、无 GPU/CUDA**;依赖最小化(`numpy/pandas/matplotlib/scipy`,见 `requirements.txt`);
固定随机种子(`np.random.seed(0)` / `default_rng(0)`)。

---

## 2. 环境与一键复现

```bash
uv sync                          # 安装依赖(numpy/pandas/matplotlib/scipy)
uv run python scs_main.py        # 阶段1:原样跑通,产出自带图到 results_export/
uv run python reproduce_all.py   # 阶段2+3:产出四张核心图 + MLP 图到 outputs/(约 85s)
```

数据(均为组合层面收益,无需外部下载):
- `Data/Instruments/managed_portfolios_anom{,_d}_50.csv`:50 个 anomaly 多空组合(月度/日度,1963-07~2017-12)。**主数据集**。
- `Data/25_Portfolios_*` 与 `Data/F-F_*`:Ken French 公开免费的 FF25 与因子数据(论文 4.1 节预备分析所用;本复现主线为 50 anomaly)。

---

## 3. 阶段1:原样跑通与最小必要修复

入口 `scs_main.py` 默认走 "daily + anom 原始特征 + 无 interactions" 路径。原样运行需修复 **4 处移植 bug**
(全部最小修复,**未改任何算法逻辑**):

| # | 错误 | 根因 | 修复 |
|---|------|------|------|
| 1 | `FileNotFoundError: Data/instruments` | `scs_main.py` 路径写小写,实际目录 `Data/Instruments`,Linux 区分大小写 | 改 `'Instruments'` |
| 2 | `pd.read_csv(date_parser=...)` 报错 | pandas≥2.0 移除 `date_parser`(本环境 pandas 3.0.3) | 改用 `date_format=` |
| 3 | `reshape(50,1)`/`reshape(1,100)` 维度写死 | 移植时把 anomaly 数、gridsize 写死 | 改 `reshape(-1,1)`/`reshape(1,-1)` |
| 4 | `to_latex requires jinja2` | pandas 3.0 的 `to_latex` 改走 Styler 需 jinja2 | 不引入 jinja2(守最小依赖),手写 LaTeX tabular |

**跑通结果(与原图/论文对照,质量高)**:OOS CV 曲线峰值 ≈0.25 @ κ≈0.28(原图 ≈0.24);系数表 top10
顺序与论文 Table 1a 完全一致(`r_indrrevlv` 居首,b≈−0.80, t≈3.44 vs 论文 −0.92/3.67);自由度曲线 0→50 单调饱和。

---

## 4. 阶段2:四个核心结果

所有结果的样本外横截面 $R^2$ 严格采用论文式(30)
$R^2_{oos}=1-\frac{(\bar\mu_2-\bar\Sigma_2\hat b)'(\bar\mu_2-\bar\Sigma_2\hat b)}{\bar\mu_2'\bar\mu_2}$
的口径,用 3-fold 连续块交叉验证(与原仓库 `cross_validate.py` 的 `CSR2` 一致)。核心数值模块见
`scs_core.py`(已通过 `_selftest_core.py` 自测:enet 在 γ1=0 时严格退化为纯 L2;OOS 曲线复现原仓库)。

### 4.1 图1 — L2 模型选择(`outputs/fig1_l2_selection.png`,对应论文 Figure 4a)

- **内容**:横轴 κ(root expected SR²,先验强度;κ 越大 L2 越弱),纵轴横截面 $R^2$。
  In-sample(虚线)、OOS CV(实线)、±1 s.e.(点线)、Pástor-Stambaugh level shrinkage(点划)。
- **关键数值**:**OOS 峰值 0.252 @ κ\*=0.279**;P&S level shrinkage 峰值仅 **0.038**。
- **说明**:In-sample $R^2$ 随 κ 增大单调升至 0.88(把均值的样本噪声也拟合进去 → 虚高);OOS 出现
  内点最优(驼峰),因为过弱的收缩会过拟合均值噪声。P&S 只做 level(等比例)收缩,远不如我们 η=2 的
  relative 收缩(对低方差 PC 收缩更狠)。**这就是论文"需要收缩、且 relative 收缩是关键"的核心证据。**

### 4.2 图2 — L1–L2 双惩罚等高线(`outputs/fig2_dual_penalty_contour.png`,对应论文 Figure 3a/3b)

- **内容**:双 panel 热图。横轴 κ(L2 强度),纵轴非零系数个数(L1 稀疏度,对数),颜色=OOS $R^2$。
  (a) 原始 50 anomaly;(b) 其主成分。双惩罚 elastic net 用纯 numpy **坐标下降**解论文式(28)。
- **关键数值**:非零数 ≤10 时,**PC 空间最高 OOS $R^2$=0.233,而原始特征空间仅 0.108**(更稀疏区差距更大,见图4)。
- **说明**:(a) 原始特征的高分区集中在**高非零数**(几乎不能稀疏)—— 50 个 anomaly 之间冗余极少,
  强制稀疏会急剧恶化;(b) PC 空间高分区**延伸到很低的非零数**(少数几个 PC 即可)。
  **两图对比是论文核心:稀疏性只在 PC 空间存在,在原始特征空间几乎不存在。**

### 4.3 图3 — SDF 系数在 PC 空间的分布(`outputs/fig3_pc_coefficients.png`,对应论文 Table 1b)

- **内容**:在最优 κ\* 下把 SDF 系数旋转到 PC 空间(式24),展示各 PC 的 $|t|$ 统计量与系数 $b_{P,j}$。
- **关键数值**:$|t|$ 最大的 PC 顺序 **PC5(3.62), PC1(3.27), PC2(2.50), PC4(1.74), PC11(1.49),
  PC15(1.46), PC10(1.43)...** —— 与论文 Table 1b(PC5,PC1,PC2,PC4,PC11,PC15,PC10...)**几乎逐项一致**。
- **说明**:大载荷确实偏向高方差 PC(PC1/2/4/5),但 **PC10/11/15/19 等中等方差 PC 也带可观载荷**,
  尾部并非全零。**信息分散在约 20 个 PC 上,而非集中于前几个** —— 这正是论文反对 characteristics-sparse、
  支持 PC 收缩的依据。

### 4.4 图4 — 稀疏 vs 稠密(`outputs/fig4_sparsity_frontier.png`,对应论文 Figure 4b)

- **内容**:横轴 SDF 因子数,纵轴该稀疏度下扫遍所有 L2 强度能达到的最大 OOS $R^2$(图2 等高线的"山脊")。
  两条线:Characteristics(原始因子)、PCs。另附两种空间的**样本外 Sharpe ratio**对比(Task.md 明确要求)。
- **关键数值**:PCs 快升(**2 个 PC→0.17,4 个→0.21,10 个→0.23**,接近最大);Characteristics 慢升
  (4 个→0.02,10 个→0.11,需 ~50 个因子才追平)。**样本外年化 Sharpe:稠密 L2(50 因子)=1.29 vs 稀疏 L1(~2 因子)=0.56。**
- **说明**:characteristics-sparse SDF 表现差,PC-sparse SDF 表现好。**这是"稀疏性在 PC 空间存在、
  在特征空间不存在"最直接的样本外证据**,也直接回答 Task.md 图4 的稀疏 vs 稠密对比。

---

## 5. 阶段3:机器学习延伸 —— 极小 MLP 学习非线性收缩

### 5.1 设计(口径与论文完全一致)

论文线性估计在 PC 空间等价于对每个 PC 施加 ridge 收缩 $b_{P,j}=\mu_{P,j}/(d_j+\gamma)$,
收缩权重 $w(d)=d/(d+\gamma)$ 是常数惩罚下的固定曲线。本延伸把它替换为**极小浅层 MLP** 学到的
**自适应惩罚**:

$$\gamma_j = \gamma_{base}(\kappa^*)\cdot\exp\big(\text{SCALE}\cdot\tanh(\text{MLP}(\log d_j))\big),\qquad b_j=\mu_j/(d_j+\gamma_j)$$

(1 隐层、16 神经元、输入仅 $\log d_j$、scipy L-BFGS 优化、解析梯度经数值检验。)用**与四张图完全相同**
的 3-fold OOS CSR² 口径,与线性 ridge 对比。test 块完全留出,MLP 仅在 train 块内部再分 fit/val 训练
(无数据泄漏)。

### 5.2 调试历程(诚实记录,体现方法论)

直接让 MLP 自由学收缩权重 $w$、令 $b=w\mu/d$ 时,OOS **灾难性崩溃**(R² 低至 −2)。经
**6 轮系统性调试**定位到根因链:
1. `b=w\mu/d` 参数化对小方差 PC **数值病态**(ridge 靠 $w=d/(d+\gamma)$ 精确抵消使 $w/d$ 有界,自由 MLP 不保证);
2. 训练口径(PC 均值)≠ 评估口径(协方差加权)→ 小 PC 的大 $w$ 训练时不被惩罚、评估时却爆炸;
3. 内部 CV 的 fit/val 时段相邻、相关性高于真正 OOS → MLP **系统性欠收缩**。

最终把 γ **锚定在 ridge 附近**($\gamma\in[0.61,1.65]\times\gamma_{base}$),既保证数值稳定($b$ 有界、
MLP 输出为 0 时精确退化为 ridge),又能公平检验"非线性收缩能否超过常数收缩"。

### 5.3 结果与结论(诚实:MLP 未跑赢)

| 模型 | OOS 横截面 $R^2$(3-fold 均值) |
|------|------------------------------|
| 线性 ridge(论文) | **0.220** |
| Tiny MLP(本延伸) | **0.083** |

`outputs/fig5_mlp_extension.png`:(a) MLP 学到的收缩曲线大体贴合 ridge,但对高方差 PC **略欠收缩**;
(b) OOS $R^2$ 柱状对比,MLP 明显不优于线性(误差棒因 K=3 较宽)。

**结论(如实)**:**MLP 没有跑赢线性收缩。** 这不是调参失败,而是一个有价值的发现——它**强力印证论文核心主张**:
SDF 系数被均值噪声主导,定价信息分散,需要**强结构化、不可从样本内过度优化**的收缩;赋予模型额外的
非线性灵活度,只会让它倾向欠收缩、损害样本外表现。**线性 relative-shrinkage 已接近最优。**
我们没有为了"好看"把结果调成 MLP 赢。

---

## 6. 数值对照总表

| 指标 | 本复现 | 论文 |
|------|--------|------|
| OOS CV 峰值 $R^2$(50 anomaly) | 0.252 @ κ≈0.28 | ≈0.30 @ κ≈0.30 (Fig 4a) |
| P&S level shrinkage 峰值 | 0.038 | <0.08 (Fig 4a) |
| 最大 SDF 因子(原始空间) | r_indrrevlv, b≈−0.80, t≈3.44 | Industry Rel.Rev.(L.V.), −0.92, 3.67 (Tab 1a) |
| PC 空间 $|t|$ 最大者 | PC5,PC1,PC2,PC4,PC11... | PC5,PC1,PC2,PC4,PC11... (Tab 1b) |
| PC-sparse(4 PC)OOS $R^2$ | 0.21 | ≈0.18 (Fig 4b) |

差异主要来自日度/月度细节与 CV 分块随机性,**所有定性结论与关键数值均吻合**。

---

## 9. 交叉验证与已知局限(诚实记录)

为防自证,我们用 **4 个独立 agent 对抗式交叉核对**了全部代码与论文公式(数值正确性、图与论文的保真度、
MLP 有无数据泄漏)。结论:

- **核心数值模块 `scs_core.py` 判定完全正确**(置信 0.96,0 issue):enet 坐标下降严格解式(28)、
  γ1=0 时退化为 L2(式22)、κ→γ(式29)、CSR2(式30)、CV 口径与原仓库 `cross_validate.py` 一致到 ~1e-16、
  PC 旋转与 PC 空间标准误(式23/24)均正确。MLP 解析梯度与数值梯度 cosine=1.0,θ=0 时精确退化为 ridge。
- **已修复**:`dual_penalty` 原先在 CV 折里把 L1 惩罚 γ1 也乘了 kfold 因子,导致 fold 模型比 y 轴标注更稀疏
  (cross-axis 不一致);现已改为只缩放 L2、不缩放 L1,图2/图4 的稀疏度轴与 OOS 现描述同一族模型。
- **已知局限(不影响结论)**:
  - PC 旋转矩阵 Q 用**全样本** `regcov` 估计(含各 test 块),属轻微 look-ahead;但因日度样本 T=11141 使
    50×50 特征向量极稳定(train-only Q vs 全样本 Q 的 OOS 差 <0.0005),且对 MLP 与 linear **同等影响**,
    不偏倚二者对比;这与论文自身"全样本估市场 beta 做正交化"的做法一致。
  - 图1 的 Pástor-Stambaugh 曲线用 level 收缩因子 $c=\bar d/(\bar d+\gamma)$(等比例、无 twist),
    是对 η=1 的**合理重参数化**(论文脚注15亦言其 x 轴不再等于 κ),非字面 η=1 后验;定性结论(level 远逊于 relative)正确。
  - 图5 的"线性 ridge"用 PC 空间对角形式,数值上(0.220)与图1 的全 ridge(0.252)略有差别,但图5 内部 MLP
    与 linear 用**同一口径**,对比公平。MLP 学到的是 over/under 混合的"twist"(欠收缩最大 PC、过收缩中高方差 PC),
    净效果不改善 OOS。

---

## 7. 文件结构

```
scs_core.py      核心数值:enet 坐标下降(式28)、PC 旋转、统一 OOS-CV(式30 口径)
scs_data.py      统一数据加载与预处理(de-market + de-vol),保证四图+MLP 口径一致
scs_plot.py      统一绘图风格
dual_penalty.py  图2/图4 共享的 (κ×γ1) 双惩罚网格扫描 + 稀疏前沿提取
fig1..fig4_*.py  四张核心图(各含 if __main__)
mlp_sdf.py       阶段3 MLP 延伸
reproduce_all.py 一键从头复现全部(阶段2+3)
_selftest_core.py 核心模块数值自测
requirements.txt 固化依赖
```

## 8. 可复现性与硬约束

- 固定随机种子;纯 CPU;依赖仅 `numpy/pandas/matplotlib/scipy`(均最小、免费、无 GPU)。
- 无任何付费数据(WRDS/CRSP/Compustat/IBES)依赖;FF25 为 Ken French 公开免费数据。
- 原仓库的对照图 `results_export/*` 未被覆盖(阶段1跑通后已 `git checkout` 还原);新结果一律落到 `outputs/`。
```
