# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目性质

本科金融计量课程的**组队复现项目**,目标是复现 Kozak, Nagel & Santosh (2020) "Shrinking the Cross-Section" (JFE)。
本仓库是 lukaskoerber 对官方 MATLAB 实现 (serhiykozak/SCS) 的 **Python 移植**,移植不完整(见下方"已知问题")。

**论文原文**:仓库根目录的 `Shrinking-the-Cross-Section.pdf` 即原文(Kozak, Nagel & Santosh 2020)。需要时用 Read 工具的 `pages` 参数**按页直接读**(原生支持 PDF,公式可正确理解)。**不要转成 md / 文本副本** —— 该论文公式密集(矩阵代数、SDF 推导),普通提取工具会让公式乱码(上下标丢失、符号错位),而高质量 OCR-to-LaTeX 工具与本项目"最小依赖、纯 CPU、无付费"约束冲突,得不偿失。

**任务全文见 `Task.md`** —— 这是用户给的权威任务简报,分阶段推进,**每阶段结束停下汇报、等确认再继续**,报告用中文。下面是必须遵守的硬约束:

- **禁止任何付费数据依赖**:没有 WRDS / CRSP / Compustat / IBES,不写需要 API key / 付费订阅 / 拉原始个股数据的代码路径。仓库自带的组合层面收益数据可用。
- **纯 CPU,禁止 GPU / CUDA**(PyTorch-GPU、cupy 等)。方法是小规模线性代数,数据约 50 列 × 数百~数千行。
- **最小化依赖**:优先 numpy / scipy / pandas / matplotlib / scikit-learn,新增库要先说明理由。
- **不要重写整个仓库**:在现有代码上做最小必要修改,除非原代码无法运行。
- **可复现性**:固定随机种子(`scs_main.py` 已 `np.random.seed(0)`),记录依赖版本。

## 环境与运行

Python 项目用 **uv** 管理(`requires-python >=3.14`)。

```bash
uv sync                       # 安装依赖(注意:见下方 pyproject 缺依赖问题)
uv run python scs_main.py     # 真正的入口:跑完整复现,出图到 results_export/
```

- **入口是 `scs_main.py`,不是 `main.py`**(`main.py` 是 uv 生成的 "Hello" 空壳,无关)。
- **无测试套件、无 lint 配置**。验证靠跑 `scs_main.py` 看是否产出图表、与 `results_export/*_original.png` 对照。
- `pyproject.toml` 的 `dependencies = []` 是**空的**,但代码实际依赖 `numpy / pandas / matplotlib`。首次跑通前需 `uv add numpy pandas matplotlib`(阶段 3 的 MLP 延伸再按需加 `scikit-learn`)。Task.md 要求把实际依赖固化(写进 `requirements.txt` 或 pyproject)。

## 代码架构(核心数据流)

论文方法:估计 SDF $M_t = 1 - \sum_i b_i (f_{i,t} - \mathbb{E}[f_i])$ 的系数向量 $b$,对因子协方差做 **L2 (ridge) 收缩** —— 等价于收缩低方差主成分(PC)的贡献。结论是 L2 才真正起作用,纯 L1 稀疏选因子表现差。

调用链:`scs_main.py` → `load_managed_portfolios` → `SCS_L2est`(内部用 `cross_validate` + `utils`):

1. **`scs_main.py`** — 配置开关(`daily=True`、`interactions=False`、`dataprovider='anom'`)、随机种子、`default_params`,在 `Data/Instruments/` 定位 managed portfolios CSV,加载后调 `SCS_L2est`。默认走 "anom 原始特征、无 interactions" 这条路径(其余分支未完成,见下)。

2. **`load_managed_portfolios.py`** — 读 CSV(列:`date`、`rme`=市场超额收益、其余为 anomaly 多空组合收益)。`omit_prefixes=['rX_','r2_','r3_']` 丢弃派生 instrument 列,丢稀疏列与含 NaN 行。返回 `(dates, re, mkt, anomalies)`。

3. **`SCS_L2est.py`** — **核心 L2 收缩估计器**。流程:de-market(回归剔除市场 beta,`utils.demarket`)→ de-vol(归一化到市场波动率)→ 按 `oos_test_date` 切 train/test → 算正则化协方差 `X=regcov(r)` 与均值 `y` → 对 `X` 做 SVD → 把惩罚 `L2pen` 与 κ(root expected SR²)互转(`kappa2pen`),在 κ 网格上逐点用 `l2est` 估 `b` 并 `cross_validate` → 取 CSR2 最大的最优 κ → 出图(自由度、系数路径、t 统计路径、CV 目标)+ 系数表。绘图/出表函数(`plot_dof`、`plot_L2coefpaths`、`plot_L2cv`、`table_L2coefs`)都在本文件内,直接写死保存到 `results_export/`。

4. **`cross_validate.py`** — 连续块 k-fold 交叉验证(默认 `kfold=3`,`cvpartition_contiguous` 按时间顺序切块,非随机)。目标函数只实现了 **`CSR2`**(横截面 R²,`bootstrp_obj_CSR2`);GLS / SR / MVU 等在 map 里但被注释掉 → 切换会 KeyError。

5. **`utils.py`** — `demarket`(beta 回归去市场)、`regcov`(带 flat-Wishart 先验的收缩协方差)、`l2est`(ridge 解 $b=(X+\lambda I)^{-1}y$,可选返回标准误)。

6. **`load_ff25.py` / `load_ff_anomalies.py`** — FF25 组合 / FF 因子加载器,**当前主流程未调用**(`dataprovider='anom'`),仅在切到 ff25 数据源时才用到。

## 已知问题 / 移植坑(跑通前很可能需要最小修复)

- **路径大小写 bug**:`scs_main.py:26` 写 `os.path.join(datapath, 'instruments')`(小写),实际目录是 `Data/Instruments`(大写)。Linux 区分大小写 → `os.listdir` 抛 `FileNotFoundError`。
- **pandas API 过时**:`load_managed_portfolios.py:18` 用了 `pd.read_csv(..., date_parser=...)`,该参数在 pandas ≥2.0 已移除,新环境会报错(需改用 `date_format=`)。
- **硬编码维度**:`SCS_L2est.py:235` 的 `d.reshape(50,1)` / `np.array(l).reshape(1,100)` 写死了"恰好 50 个 anomaly"和"gridsize=100"。改 anomaly 数或网格大小会崩。
- **未完成分支**:`scs_main.py` 的 `ff25` 分支是 `pass` 空操作;`SCS_L2est` 里 `demarket_conditionally` / `devol_conditionally` 调用的 `demarketcond` / `devolcond` 未定义(只有 unconditionally 路径可用)。改默认开关前先确认对应分支已实现。
- **输出目录**:现有图固定写到 `results_export/`(已有 `*_original.png` 作为论文原图对照)。Task.md 阶段 2 要求新结果存到 `outputs/`,新增图请按任务要求落到 `outputs/`,不要覆盖 `results_export/` 的对照图。
