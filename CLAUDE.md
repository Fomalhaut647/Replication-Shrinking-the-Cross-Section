# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目性质

本科金融计量课程的**组队复现项目**(北京大学,第三组;成员姓名/学号见 `report/authors.local.tex`,该文件已 gitignore 不入公开仓库),目标是复现 Kozak, Nagel & Santosh (2020) "Shrinking the Cross-Section" (JFE)。
本仓库源自 lukaskoerber 对官方 MATLAB 实现 (serhiykozak/SCS) 的 **Python 移植**。

**当前状态(2026)**:移植 bug 已修、原样跑通;论文**四个核心结果已复现**(fig1-4)、一个 **MLP 机器学习延伸**(fig5)、**五项扩展**(R1 FF25、R2 去 look-ahead、R3 bootstrap CI、R4 字面 η=1、EXT 广义收缩曲线;fig6-8)均已完成并验证;`reproduce_all.py` 约 2 分钟一键复现。**最终课程报告 `report/report.pdf`(11 页中文)为权威交付物**,提交截止 2026-06-23。

**论文原文**:仓库根目录的 `Shrinking-the-Cross-Section.pdf` 即原文(Kozak, Nagel & Santosh 2020)。需要时用 Read 工具的 `pages` 参数**按页直接读**(原生支持 PDF,公式可正确理解)。**不要转成 md / 文本副本** —— 该论文公式密集(矩阵代数、SDF 推导),普通提取工具会让公式乱码(上下标丢失、符号错位),而高质量 OCR-to-LaTeX 工具与本项目"最小依赖、纯 CPU、无付费"约束冲突,得不偿失。

**任务全文见 `docs/Task.md`** —— 这是用户给的权威任务简报,分阶段推进,**每阶段结束停下汇报、等确认再继续**,报告用中文。下面是必须遵守的硬约束:

- **禁止任何付费数据依赖**:没有 WRDS / CRSP / Compustat / IBES,不写需要 API key / 付费订阅 / 拉原始个股数据的代码路径。仓库自带的组合层面收益数据可用。
- **纯 CPU,禁止 GPU / CUDA**(PyTorch-GPU、cupy 等)。方法是小规模线性代数,数据约 50 列 × 数百~数千行。
- **最小化依赖**:优先 numpy / scipy / pandas / matplotlib / scikit-learn,新增库要先说明理由。
- **不要重写整个仓库**:在现有代码上做最小必要修改,除非原代码无法运行。
- **可复现性**:固定随机种子(`scs_main.py` 已 `np.random.seed(0)`),记录依赖版本。

## 环境与运行

Python 项目用 **uv** 管理(`requires-python >=3.14`)。依赖已固化(`requirements.txt` / `uv.lock`:`numpy / pandas / matplotlib / scipy`)。

```bash
uv sync                          # 安装依赖
uv run python scs_main.py        # 阶段1:原样跑通(原仓库 SCS_L2est 路径),出图到 results_export/
uv run python reproduce_all.py   # 阶段2/3 + 五项扩展:一键复现 fig1-8 到 outputs/(约 2 分钟)
uv run python _selftest_core.py  # 核心数值自测(6 项,含一般-η 守卫)
```

- **两个入口**:`scs_main.py`(阶段1 原样跑通,原仓库 L2 估计器路径)与 **`reproduce_all.py`(论文复现主线:核心四图 + MLP + 五项扩展)**。`main.py` 是 uv 生成的空壳,无关。
- **最终课程报告**:`report/report.pdf`(11 页中文 LaTeX;源 `report/report.tex`,`xelatex` 本地可编译。字体用已装 **AR PL UMing CN 宋体 + Droid Sans Fallback**,非 fandol——fandol 未装会编译失败)。
- **无 lint 配置**。验证靠 `reproduce_all.py` 产出 fig1-8 + `_selftest_core.py` 6 项自测 + 与 `results_export/*_original.png` 对照。
- `requirements.txt` 已固化实际依赖;`pyproject.toml` 原 `dependencies = []` 空依赖的坑已解决(阶段1)。

## 代码架构(核心数据流)

论文方法:估计 SDF $M_t = 1 - \sum_i b_i (f_{i,t} - \mathbb{E}[f_i])$ 的系数向量 $b$,对因子协方差做 **L2 (ridge) 收缩** —— 等价于收缩低方差主成分(PC)的贡献。结论是 L2 才真正起作用,纯 L1 稀疏选因子表现差。

**两条代码路径**(互不覆盖):

**A. 复现主线 + 五项扩展(`reproduce_all.py`,出图 `outputs/fig1-8`)—— 主要工作在这里**:统一纯函数核心,口径一致。
- **`scs_core.py`** — 核心数值:enet 坐标下降(式28)、PC 旋转、CSR2(式30)、`oos_cv_r2`;**统一一般-η OOS-CV**(`precompute_folds_eta` + `eta_oos`,收缩权重 $w_j=d_j^{\eta-1}/(d_j^{\eta-1}+\gamma)$,η=2 精确退化纯 L2 ridge,**R2/R4/EXT 共用此基座**,每折 train-only 旋转)。
- **`scs_data.py`** — 统一数据口径(de-market + de-vol):`prepared()`(50 anomaly)与 `prepared_ff25()`(扩展 R1)。
- **`dual_penalty.py`** — 双惩罚(elastic net)网格扫描 + 稀疏前沿;`pc_trainQ=True` 选项供 R2 用 train-only Q。
- **`fig1-4_*.py`** 四核心图(fig1 含 R4 字面 η=1 P&S);**`mlp_sdf.py`** 图5 MLP 延伸;**`gen_shrinkage.py`** 图7/EXT 广义收缩曲线(学 η);**`r2_lookahead.py`** 图8/R2 去 look-ahead;**`bootstrap_ci.py`** R3 分块 bootstrap CI;**`fig6_ff25_sparsity.py`** 图6/R1 FF25。
- **`_selftest_core.py`** 6 项核心自测(含 `eta_oos(η=2)==ridge` 守卫)。

**B. 阶段1 原样跑通(`scs_main.py`,原仓库路径,出图 `results_export/`)**:`scs_main.py` → `load_managed_portfolios` → `SCS_L2est`(内部用 `cross_validate` + `utils`),已修 4 处移植 bug(见下)。各文件:

1. **`scs_main.py`** — 配置开关(`daily=True`、`interactions=False`、`dataprovider='anom'`)、随机种子、`default_params`,在 `Data/Instruments/` 定位 managed portfolios CSV,加载后调 `SCS_L2est`。默认走 "anom 原始特征、无 interactions" 这条路径(其余分支未完成,见下)。

2. **`load_managed_portfolios.py`** — 读 CSV(列:`date`、`rme`=市场超额收益、其余为 anomaly 多空组合收益)。`omit_prefixes=['rX_','r2_','r3_']` 丢弃派生 instrument 列,丢稀疏列与含 NaN 行。返回 `(dates, re, mkt, anomalies)`。

3. **`SCS_L2est.py`** — **核心 L2 收缩估计器**。流程:de-market(回归剔除市场 beta,`utils.demarket`)→ de-vol(归一化到市场波动率)→ 按 `oos_test_date` 切 train/test → 算正则化协方差 `X=regcov(r)` 与均值 `y` → 对 `X` 做 SVD → 把惩罚 `L2pen` 与 κ(root expected SR²)互转(`kappa2pen`),在 κ 网格上逐点用 `l2est` 估 `b` 并 `cross_validate` → 取 CSR2 最大的最优 κ → 出图(自由度、系数路径、t 统计路径、CV 目标)+ 系数表。绘图/出表函数(`plot_dof`、`plot_L2coefpaths`、`plot_L2cv`、`table_L2coefs`)都在本文件内,直接写死保存到 `results_export/`。

4. **`cross_validate.py`** — 连续块 k-fold 交叉验证(默认 `kfold=3`,`cvpartition_contiguous` 按时间顺序切块,非随机)。目标函数只实现了 **`CSR2`**(横截面 R²,`bootstrp_obj_CSR2`);GLS / SR / MVU 等在 map 里但被注释掉 → 切换会 KeyError。

5. **`utils.py`** — `demarket`(beta 回归去市场)、`regcov`(带 flat-Wishart 先验的收缩协方差)、`l2est`(ridge 解 $b=(X+\lambda I)^{-1}y$,可选返回标准误)。

6. **`load_ff25.py` / `load_ff_anomalies.py`** — FF25 组合 / FF 因子加载器。`load_ff25` 已修 2 处移植 bug(见下),现由 `scs_data.prepared_ff25()` / `fig6`(扩展 R1)调用;原 `scs_main.py` 的 ff25 分支仍是 pass(R1 不走 scs_main,走 reproduce_all)。

## 已知问题 / 移植坑(原仓库坑;阶段1 与 R1 已修复,记录备查)

- **路径大小写 bug** ✅已修:`scs_main.py` 原写 `'instruments'`(小写),实际目录 `Data/Instruments`(大写),Linux 区分大小写。已改 `'Instruments'`。
- **pandas API 过时** ✅已修:`load_managed_portfolios.py` 的 `date_parser=` 在 pandas≥2.0 已移除(本环境 pandas 3),已改 `date_format=`。
- **硬编码维度** ✅已修:`SCS_L2est.py` 的 `reshape(50,1)`/`reshape(1,100)` 写死了 anomaly 数与网格大小,已改 `reshape(-1,1)`/`reshape(1,-1)`。
- **`load_ff25.py` 两处 bug** ✅已修(R1):① 因子列实际名 `Mkt-RF`(连字符)原写 `Mkt_RF`(下划线)→ KeyError;② `DataFrame - Series` 减 RF 会按列名对齐成全 NaN,已改 `.sub(..., axis=0)` 按行减。这俩之所以一直没暴露是 ff25 分支从没真正跑过。
- **未完成分支(仍存在)**:`SCS_L2est` 的 `demarket_conditionally`/`devol_conditionally` 调用的 `demarketcond`/`devolcond` 未定义(只有 unconditionally 路径可用);`scs_main.py` 的 ff25 分支仍是 pass(R1 改走 `reproduce_all` 路径,不依赖它)。
- **输出目录约定**:`results_export/`(阶段1 原图 + `*_original.png` 对照)**不要覆盖**;复现主线与扩展的新图一律落 `outputs/`(fig1-8)。这一约定全程遵守。
