# 交付报告:Shrinking the Cross-Section 复现

> Kozak, Nagel & Santosh (2020), *Journal of Financial Economics* 135(2) 复现项目交付报告。
> 技术细节与每张图的完整中文解释见 [`REPLICATION_REPORT.md`](REPLICATION_REPORT.md);本报告聚焦
> **交付概览、验收对照、运行方式与质量保证**。

> **⚠️ 权威版本提示(2026 更新)**:本文记录的是早期阶段(核心四图 + 单个 MLP 延伸)的交付概览。最终课程
> 交付物为 [`report/report.pdf`](../report/report.pdf)(11 页中文 LaTeX),在此之上新增**五项扩展**(R1-R4 + EXT)
> 与 bootstrap CI,并更正了 look-ahead 与 MLP 的旧数值(详见 `REPLICATION_REPORT.md` 顶部提示)。一键复现
> `reproduce_all.py` 现产出 fig1-8(约 2 分钟)。

---

## 一、一句话总结

在 lukaskoerber 的 Python 移植版基础上,**让仓库在纯 CPU、当前环境(Python 3.14 + pandas 3)下一键跑通**,
**复现了论文四个核心结果**,并完成一个**诚实的机器学习延伸**(极小 MLP vs 线性收缩)。全程无付费数据、无 GPU、
固定随机种子,经 4 个独立 agent 对抗式验证。

---

## 二、Task.md 验收标准对照

| 验收标准 | 状态 | 证据 |
|---------|------|------|
| 阶段0 仓库探索汇报、核对假设 | ✅ | 见会话报告(入口=`scs_main.py`、数据=50 anomaly、无付费依赖) |
| 纯 CPU、当前环境一键跑通,依赖写进 `requirements.txt` | ✅ | `uv run python reproduce_all.py`(~90s);`requirements.txt` 已固化 |
| 四个核心结果生成并存于 `outputs/`,各有文字解释 | ✅ | `outputs/fig1–4.png` + `REPLICATION_REPORT.md` 第4节 |
| 阶段3 延伸完成,结果诚实(赢或不赢都接受) | ✅ | `outputs/fig5.png`;MLP 0.083 vs ridge 0.220,**如实未跑赢** |
| 全程未引入付费数据、未引入 GPU 依赖 | ✅ | 仅 `numpy/pandas/matplotlib/scipy`;数据为组合收益 + Ken French 公开数据 |
| 从头可复现的脚本,固定随机种子 | ✅ | `reproduce_all.py`(`np.random.seed(0)` / `default_rng(0)`) |

---

## 三、如何运行

```bash
uv sync                          # 安装依赖(numpy/pandas/matplotlib/scipy)
uv run python scs_main.py        # 阶段1:原样跑通,自带图 -> results_export/
uv run python reproduce_all.py   # 阶段2+3:四张核心图 + MLP -> outputs/(约 90s)
uv run python _selftest_core.py  # 核心数值模块自测(可选)
```

---

## 四、核心成果

| 图 | 论文对应 | 关键数值(复现) | 一句话结论 |
|----|---------|---------------|-----------|
| `fig1_l2_selection` | Figure 4a | OOS 峰 **0.252 @ κ\*=0.279**;P&S level=0.038 | 需收缩,且 relative≫level |
| `fig2_dual_penalty_contour` | Figure 3a/3b | ≤10 非零:PC=0.233 vs char=0.108 | 稀疏性只在 PC 空间存在 |
| `fig3_pc_coefficients` | Table 1b | \|t\| 序 PC5,PC1,PC2,PC4,PC11… 与论文一致 | 信息分散在多个 PC |
| `fig4_sparsity_frontier` | Figure 4b | PC 4因子→0.21,char→0.02;Sharpe 1.29 vs 0.56 | char-sparse 不成立 |
| `fig5_mlp_extension` | (延伸) | MLP **0.083** vs ridge **0.220** | MLP 未跑赢,印证线性近最优 |

**总结论**:完整复现"L2 收缩有效、纯 L1 稀疏表现差、稀疏性只在 PC 空间存在";机器学习延伸**诚实地表明**
赋予收缩函数额外非线性灵活度无法改善样本外表现(反因均值噪声倾向欠收缩),呼应论文"信息分散、线性收缩已近最优"。

---

## 五、质量保证

1. **核心数值自测** `_selftest_core.py`:enet 在 γ1=0 时严格退化为纯 L2、OOS 曲线复现原仓库口径。
2. **4 个独立 agent 对抗式验证**:
   - 核心模块 `scs_core.py` 判定**完全正确**(置信 0.96,0 issue),与论文式(22)(28)(29)(30) + 原仓库
     `cross_validate.py` 一致到 ~1e-16。
   - 据验证发现**修复 1 个 medium issue**(L1 惩罚误随 kfold 缩放 → 已改为只缩放 L2)。
   - MLP 解析梯度与数值梯度 cosine=1.0;θ=0 时精确退化 ridge;SCALE 扫描证明"越自由越差",结论稳健未作弊。
3. **已知局限**(诚实记录,均不影响结论):PC 旋转用全样本 Q 的轻微 look-ahead、P&S 曲线为 level 收缩的合理
   重参数化、fig5 线性基线用 PC 空间对角 ridge。详见 `REPLICATION_REPORT.md` 第9节。

---

## 六、文件结构

```
scs_main.py            阶段1 入口(已修复移植 bug)
scs_core.py            核心数值:enet 坐标下降(式28)、PC 旋转、统一 OOS-CV(式30)
scs_data.py            统一数据加载与预处理(口径一致)
scs_plot.py            统一绘图风格
dual_penalty.py        图2/图4 共享的双惩罚网格扫描 + 稀疏前沿
fig1..fig4_*.py        四张核心图
mlp_sdf.py             阶段3 MLP 延伸
reproduce_all.py       一键复现(阶段2+3)
_selftest_core.py      核心模块自测
requirements.txt       依赖固化
docs/                  REPLICATION_REPORT.md(技术详解) / DELIVERY_REPORT.md(本报告) / Task.md(任务简报)
outputs/               全部复现图与中间数据
```

---

## 七、硬约束遵守

- **无付费数据**:仅用仓库自带组合层面收益 + Ken French 公开免费数据;无 WRDS/CRSP/Compustat/IBES。
- **纯 CPU**:无 GPU/CUDA;方法为小规模线性代数 + 极小 MLP(scipy L-BFGS)。
- **最小依赖**:`numpy/pandas/matplotlib/scipy`(未引入 jinja2、sklearn 等)。
- **可复现**:固定随机种子,依赖版本锁定(`uv.lock` / `requirements.txt`)。
- **不破坏原仓库**:阶段1 仅 4 处最小移植修复;原对照图 `results_export/*` 保持原样,新结果落到 `outputs/`。
