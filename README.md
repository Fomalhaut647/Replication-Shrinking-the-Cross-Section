# Replication: Shrinking the Cross-Section

This repository hosts the replication of findings from the paper titled *Shrinking the Cross-Section* by **S. Kozak, S. Nagel, and S. Santosh**, published in the Journal of Financial Economics in 2020. The paper explores the construction of a robust stochastic discount factor (SDF) that encapsulates the collective explanatory power of numerous cross-sectional stock return predictors through an economically substantiated prior on SDF coefficients, providing a robust out-of-sample performance in high-dimensional settings.

[Link to the original paper on SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2945663).

## Abstract:
> We construct a robust stochastic discount factor (SDF) that summarizes the joint explanatory power of a large number of cross-sectional stock return predictors. Our method achieves robust out-of-sample performance in this high-dimensional setting by imposing an economically motivated prior on SDF coefficients that shrinks the contributions of low-variance principal components of the candidate factors. While empirical asset pricing research has focused on SDFs with a small number of characteristics-based factors --- e.g., the four- or five-factor models discussed in the recent literature --- we find that such a characteristics-sparse SDF cannot adequately summarize the cross-section of expected stock returns. However, a relatively small number of principal components of the universe of potential characteristics-based factors can approximate the SDF quite well.

## Dataset and Original Code:
For datasets and code, please see [Serhiy Kozak's webpage](https://www.serhiykozak.com/data).


## Repository Structure:
- **scs_main.py**: Run scs_main.py to obtain the replication results.
- **Data**: The dataset referenced in the paper.

# Results
Comparison of original results from the paper and replication results from the Python script, aligned side-by-side for comparison.

## L2 Model Selection
### In-sample and Out-of-Sample Cross-Sectional R^2 Analysis
> This section displays the original and replicated plots that present the in-sample cross-sectional R^2 (dashed line), out-of-sample (OOS) cross-sectional R^2 based on cross-validation (solid line), and OOS cross-sectional R^2 based on the proportional shrinkage (dash-dot line) as per Pástor and Stambaugh (2000).

#### Original
![Original L2 Model Selection Plot](results_export/cross_validation_original.png)

#### Replicated
![Replicated L2 Model Selection Plot](results_export/cross_validation.png)


### Largest SDF factors
> Coefficient estimates and absolute t-statistics at the optimal value of the prior root expected SR2 (based on cross-validation), with focus on the original 50 anomaly portfolios. Coefficients are sorted descending on their absolute t-statistic values.

#### Original
![Original Coefficients Table](results_export/coefficients_table_original.png)

#### Replicated
![Replicated Coefficients Table](results_export/coefficients_table_replication.png)

### Degrees of Freedom
> TODO

#### Original
![Original Degrees of Freedom Plot](results_export/degrees_of_freedom_original.png)

#### Replicated
![Replicated Degrees of Freedom Plot](results_export/degrees_of_freedom.png)

## SDF Coefficients

### Coefficient Paths
> TODO

#### Original
![Original Coefficient Paths Plot](results_export/coefficients_paths_original.png)

#### Replicated
![Replicated Coefficient Paths Plot](results_export/coefficients_paths.png)

### t-Statistic Paths
> TODO

#### Original
![Original Coefficient Paths Plot](results_export/tstats_paths_original.png)

#### Replicated
![Replicated Coefficient Paths Plot](results_export/tstats_paths.png)

---

## 复现扩展(2026):核心结果 + 机器学习延伸

在原仓库 L2 复现之上,本次工作补全了论文的**四个核心结果**并加入一个**机器学习延伸**,
全部纯 CPU、最小依赖、固定随机种子。详见 [`docs/REPLICATION_REPORT.md`](docs/REPLICATION_REPORT.md)(技术详解)
与 [`docs/DELIVERY_REPORT.md`](docs/DELIVERY_REPORT.md)(交付报告)。

```bash
uv run python reproduce_all.py   # 一键复现,产出全部图到 outputs/(约 85s)
```

| 图 | 论文对应 | 文件 |
|----|----------|------|
| L2 模型选择(IS/OOS/±se + P&S 对照) | Figure 4a | `outputs/fig1_l2_selection.png` |
| L1–L2 双惩罚 OOS R² 等高线(raw + PC) | Figure 3a/3b | `outputs/fig2_dual_penalty_contour.png` |
| SDF 系数在 PC 空间的分布 | Table 1b | `outputs/fig3_pc_coefficients.png` |
| 稀疏 vs 稠密 OOS R²(+ Sharpe 对比) | Figure 4b | `outputs/fig4_sparsity_frontier.png` |
| **MLP 延伸**:非线性自适应收缩 vs 线性 ridge | — | `outputs/fig5_mlp_extension.png` |

**核心结论**:复现了"L2 收缩有效、稀疏性只在 PC 空间存在"的全部定性结果;MLP 延伸**诚实地未跑赢**
线性收缩(OOS R² 0.083 vs 0.220),印证论文"信息分散、线性收缩已近最优"的主张。

## 仓库结构

```text
入口    scs_main.py(阶段1 原样跑通)   reproduce_all.py(阶段2+3 一键复现)
框架    scs_core / scs_data / scs_plot / dual_penalty
出图    fig1-4_*.py   mlp_sdf.py        自测  _selftest_core.py
原仓库  SCS_L2est / cross_validate / utils / load_*.py
文档    README.md   docs/(REPLICATION_REPORT · DELIVERY_REPORT · Task)
数据    Data/    产物  outputs/    原始对照图  results_export/
```


