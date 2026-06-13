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

## 复现扩展(2026):核心结果 + 机器学习延伸 + 五项扩展

在原仓库 L2 复现之上,本次工作复现论文**四个核心结果**、加入一个**机器学习延伸**,并新增**五项扩展与
稳健性分析**,全部纯 CPU、最小依赖、固定随机种子。
**最终课程报告(权威、含全部解读与分工)见 [`report/report.pdf`](report/report.pdf)**
(LaTeX 源 `report/report.tex`,中文,`xelatex` 本地可编译)。

```bash
uv run python reproduce_all.py   # 一键复现,产出 fig1-8 到 outputs/(约 2 分钟)
uv run python _selftest_core.py  # 核心数值自测(6 项,含一般-η 守卫)
```

| 图 | 论文对应 / 扩展 | 文件 |
|----|----------|------|
| L2 模型选择(IS/OOS/±se + **字面 η=1 对照**) | Figure 4a / **R4** | `outputs/fig1_l2_selection.png` |
| L1–L2 双惩罚 OOS R² 等高线(raw + PC) | Figure 3a/3b | `outputs/fig2_dual_penalty_contour.png` |
| SDF 系数在 PC 空间的分布 | Table 1b | `outputs/fig3_pc_coefficients.png` |
| 稀疏 vs 稠密 OOS R²(+ Sharpe 对比) | Figure 4b | `outputs/fig4_sparsity_frontier.png` |
| **MLP 延伸**:非线性自适应收缩 vs 线性 ridge | 机器学习延伸 | `outputs/fig5_mlp_extension.png` |
| **广义收缩曲线**:学收缩形状 η(配 bootstrap CI) | **扩展 EXT** | `outputs/fig7_generalized_shrinkage.png` |
| **去 look-ahead**:train-only Q vs 全样本 Q | **扩展 R2** | `outputs/fig8_r2_lookahead.png` |
| **FF25 预备分析**:低维性 + 稀疏 vs 稠密(负对照) | **扩展 R1** | `outputs/fig6_ff25_sparsity.png` |

**核心结论**:复现"L2 收缩有效、相对收缩是关键、稀疏性只在 PC 空间存在"的全部定性结果;机器学习
两条延伸(自由 MLP 与可学收缩形状 η)**均诚实地未取得统计显著的样本外提升**——EXT 学到的 η*≈3.6
相对论文固定 η=2 的增益 +0.04 落在 bootstrap 95% CI 内(含 0),印证"信息分散、线性相对收缩已近最优"。
扩展 R2 进一步表明纯 L2 的 PC 旋转 look-ahead 严格为 0、PC-sparse 结果未被抬高;R3 给出 η=2 的 OOS
0.252 的 95% CI [0.014, 0.377](显著为正)。

## 仓库结构

```text
入口    scs_main.py(阶段1 原样跑通)   reproduce_all.py(核心四图 + MLP + 五项扩展,一键复现)
框架    scs_core(核心数值,含统一一般-η OOS-CV) / scs_data(含 prepared_ff25) / scs_plot / dual_penalty(含 train-Q 选项)
出图    fig1-4_*.py  mlp_sdf.py(图5)  gen_shrinkage.py(图7/EXT)  r2_lookahead.py(图8/R2)  fig6_ff25_sparsity.py(图6/R1)  bootstrap_ci.py(R3)
自测    _selftest_core.py(6 项)
原仓库  SCS_L2est / cross_validate / utils / load_*.py
文档    README.md   report/(report.tex/pdf,最终报告)   docs/(REPLICATION_REPORT · DELIVERY_REPORT · Task)
数据    Data/    产物  outputs/    原始对照图  results_export/
```


