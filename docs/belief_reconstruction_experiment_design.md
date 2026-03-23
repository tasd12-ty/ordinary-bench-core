# 场景信念重建实验：论文叙事与实验设计

本文档是 §6.3 "Scene Belief Reconstruction" 实验节的完整论证设计。

---

## Part I: 论证逻辑

### 为什么需要这个实验

§4 定义了重建管线（方法论），但方法论本身不产生 finding。§6.3 要用这个管线回答一个具体的实验问题：

> VLM 对一个场景的全部空间回答，是否能够组装成一个一致的、接近真实的场景布局？

这个问题的答案直接支撑论文的核心主张——单题正确率高估了空间能力。如果一个 VLM 在 QRR 上答对了 56%，但用它的全部回答去重建场景时，重建出来的布局与真实场景严重偏离甚至无法重建，那么 56% 的准确率就是一个虚假的信号。

### 两路对比，不是三路

只需要两种重建：

**GT 重建**：用所有问题的 ground truth 答案作为约束，重建场景。这是管线验证和理论上界。如果 GT 约束下重建失败，问题出在管线或场景本身，与模型无关。

**Belief 重建**：用 VLM 的全部回答（不论对错）作为约束，重建场景。这是 VLM 外显空间信念的直接体现。VLM 自己不知道哪些回答对、哪些错，它的全部回答整体构成了它对这个场景的"信念"。

不做 Correct-only 重建。VLM 不知道自己哪些答对了，挑选正确回答做重建既不是 VLM 看见的世界，也不是真实世界，没有清晰的解释意义。

### 术语边界

我们重建出来的不是模型的内部表征（internal representation），而是由回答诱导出来的外显空间信念（externalized spatial belief）。对于黑盒 API 模型，我们无法访问其内部神经状态。我们能做的，类似行为心理学中的心理物理学实验——通过被试的行为反应推断其感知结构，而不是直接读取大脑。论文正文中需要始终使用 "externalized spatial belief" 这一表述，并在 Discussion 中讨论这一局限。

---

## Part II: §6.3 论文正文（段落式论证草稿）

### §6.3 Scene Belief Reconstruction

Question-level accuracy measures how often a model's individual answers match the ground truth, but it cannot reveal whether those answers, taken together, describe a coherent spatial world. A model might correctly answer 56\% of QRR probes and yet produce a set of pairwise distance comparisons that are mutually contradictory---no single arrangement of objects could satisfy them all simultaneously. To expose this gap between local accuracy and global coherence, we apply the scene belief reconstruction pipeline (§4) to every evaluated scene, converting each model's complete set of responses into spatial constraints and attempting to recover the implied 2D layout.

#### Pipeline validation under ground-truth constraints

We first establish that the reconstruction pipeline itself is sound by running it on ground-truth constraints: every probe is answered with its correct ground-truth value, yielding a complete and consistent set of ordinal constraints for each scene. Across all 700 scenes and all complexity levels ($n=4$ to $n=10$), the solver recovers layouts with CSR$_\text{QRR} > 0.99$, NRMS $< 0.02$, Kendall $\tau > 0.98$, and $K_\text{geom} = 1$. This confirms that any degradation observed when using model predictions can be attributed entirely to errors or inconsistencies in those predictions, not to limitations of the optimization procedure.

#### Belief reconstruction: assembling the model's spatial world

We then reconstruct each scene using the model's actual responses to all probes---correct and incorrect alike---in what we term *belief mode*. This is the central experiment of the section, because it reveals the spatial world that the model's answers collectively describe, regardless of which individual answers happen to be right. The model itself has no access to correctness labels; its full response vector is the only externally observable signal of its spatial belief, and it is this signal that we reconstruct.

Table 2 reports aggregate reconstruction metrics across scenes for each evaluated model. The contrast with the ground-truth upper bound is stark. While GT constraints produce feasible, accurate reconstructions in every scene, belief reconstruction succeeds (CSR $> 0.95$) in only X\% of scenes for the strongest model. Among scenes where reconstruction is feasible, NRMS rises from near-zero under GT to X.XX under belief, and Kendall $\tau$ drops from near-unity to X.XX, indicating that even when the model's answers are not outright contradictory, the implied layout is substantially distorted relative to the true scene.

#### Feasibility breakdown: why reconstruction fails

Not all failures are alike. We decompose belief-mode outcomes into four categories based on the reconstruction status:

*Infeasible* scenes are those in which the model's responses contain irreconcilable contradictions---cycles in the distance poset (e.g., $d(A,B) < d(C,D)$, $d(C,D) < d(E,F)$, $d(E,F) < d(A,B)$) or incompatible angular sectors in TRR. These are not merely wrong answers; they are answers that cannot coexist in any spatial arrangement. The fraction of scenes rendered infeasible by model responses, despite being fully reconstructable under GT constraints, is a direct measure of the model's structural self-contradiction.

*Underconstrained* scenes arise when too many responses are missing or malformed to pin down a unique layout, leaving the solver with multiple qualitatively different solutions ($K_\text{geom} > 1$, high spread).

*Distorted* scenes are those where the belief reconstruction is feasible and unique, but the recovered layout deviates substantially from the ground truth (high NRMS, low Kendall $\tau$). These represent a model that maintains internal consistency but whose spatial model is systematically biased.

*Faithful* scenes are the rare cases where belief reconstruction closely matches the ground truth: the model's answers, despite some individual errors, collectively imply a layout that is geometrically close to reality.

We find that X\% of scenes fall into the infeasible category for Model A versus Y\% for Model B, even though both models achieve similar per-question accuracy. This divergence---invisible to conventional accuracy reporting---reveals that the two models fail in qualitatively different ways: one produces spatially incoherent responses, while the other maintains a distorted but self-consistent spatial model.

#### The gap between accuracy and spatial coherence

To directly test whether per-question accuracy predicts reconstruction quality, we plot per-scene QRR accuracy against belief NRMS in Figure 7. If accuracy were a sufficient indicator of spatial competence, we would expect a tight negative correlation: higher accuracy should reliably yield lower NRMS. Instead, we observe substantial scatter. Scenes with identical QRR accuracy can differ by a factor of X in NRMS, and some scenes with moderate accuracy (50--60\%) yield better reconstructions than scenes with higher accuracy (70\%+). This is because accuracy treats all errors as equivalent, while reconstruction is sensitive to the *pattern* of errors: a few contradictory answers can render a scene infeasible, whereas many wrong-but-consistent answers may merely shift the layout.

This finding is the empirical core of our argument: per-question accuracy is a lossy summary that discards the structural relationships among answers, and it is precisely these relationships that determine whether a model's spatial belief forms a coherent world.

#### Controlled accuracy sweep: how much accuracy is enough?

The preceding analysis uses real model outputs, where accuracy, error patterns, and scene complexity are confounded. To isolate the effect of accuracy on reconstructability, we perform a controlled experiment: starting from ground-truth answers, we corrupt a fraction $\eta$ of responses by replacing them with uniformly random values, and measure how reconstruction quality degrades as $\eta$ increases from 0 to 0.5.

Figure 5 shows the result, stratified by scene complexity ($n$). Two patterns emerge. First, there is no single accuracy threshold; the critical corruption rate depends strongly on scene complexity. Scenes with $n=4$ objects (3 QRR + 24 TRR probes) have minimal constraint redundancy, and reconstruction breaks down sharply at $\eta \approx$ X. Scenes with $n=10$ (630 QRR + 720 TRR probes) benefit from massive redundancy and degrade more gracefully, tolerating corruption rates up to $\eta \approx$ Y before feasibility drops below 50\%. Second, the degradation is not linear: there is a phase-transition-like regime where feasibility and Kendall $\tau$ drop precipitously over a narrow range of $\eta$, suggesting that the constraint system transitions abruptly from overdetermined to underdetermined.

These curves provide a reference frame for interpreting real model performance: if a model achieves X\% QRR accuracy on $n$-object scenes, the corruption sweep tells us whether that accuracy level is above or below the reconstruction threshold for that complexity class.

#### Visualizing the belief world

Figure 2 presents side-by-side comparisons of ground-truth layouts and belief reconstructions for selected scenes spanning the complexity range. Each panel shows the top-down positions of objects after Procrustes alignment. In the 4-object scene, the belief reconstruction preserves the rough spatial arrangement but introduces rotational distortion. In the 7-object scene, the model's contradictory responses render reconstruction infeasible entirely---there is no layout consistent with its answers. In the 10-object scene, reconstruction succeeds but with severe distortion: the ordinal distance structure (Kendall $\tau$ = X.XX) is partially preserved, but the metric layout (NRMS = X.XX) bears little resemblance to the original.

These visualizations make concrete what the aggregate statistics describe: the spatial world implied by a model's answers is often a distorted, fragmented, or impossible version of reality.

---

## Part III: 可视化规格

### Figure 2: Reconstruction Cascade

- 3 行（n=4, n=7, n=10 各选一个代表性场景）× 2 列（GT 位置 / Belief 重建）
- 每格：Procrustes 对齐后的 top-down 点图，物体保留颜色和形状属性
- GT 列叠加浅色 ghost 位置作为参考
- infeasible 场景标注 "infeasible (cycle detected)" 或类似说明
- 每格右下标注 NRMS 和 Kendall τ

### Figure 5: Phase Transition

- 3 面板横向排列：n=4 / n=7 / n=10
- X 轴: corruption rate η (0 ~ 0.5)
- Y 轴（双轴或两条线）: feasible rate (%) + Kendall τ
- 带 error bar（多次随机试验的 std）
- 标注 50% feasibility 处的 η 值

### Figure 6: Feasibility Status Distribution

- 堆叠柱状图，按模型分组
- 4 种颜色：faithful / distorted / underconstrained / infeasible
- 上方标注 per-question accuracy 供对比

### Figure 7: Calibration Scatter

- X 轴: per-scene QRR accuracy
- Y 轴: belief NRMS（对数刻度可能更清晰）
- 点颜色: feasibility status（green/yellow/red）
- 可选叠加回归线或 LOWESS 曲线展示趋势

### Table 2: GT vs Belief Reconstruction

```
Model          Mode     Feasible%  CSR_QRR  Kendall τ  NRMS   K_geom
─────────────────────────────────────────────────────────────────────
Model A        GT        100.0     0.998    0.98       0.01   1.0
               Belief     XX.X     X.XXX    X.XX       X.XX   X.X
Model B        GT        100.0     0.998    0.98       0.01   1.0
               Belief     XX.X     X.XXX    X.XX       X.XX   X.X
```

---

## Part IV: 实现规格

### 新建文件

| 文件 | 职责 |
|------|------|
| `VLM-test/analysis/belief_metrics.py` | 共享工具：GT 评分合成、cross 指标、错误分类 |
| `VLM-test/analysis/belief_vs_gt.py` | GT vs Belief 对比主脚本，输出 Table 2 + Fig 6/7 数据 |
| `VLM-test/analysis/corruption_sweep.py` | 受控准确率实验，输出 Fig 5 数据 |
| `VLM-test/analysis/visualize_belief_comparison.py` | 论文 Figure 生成（Fig 2, 5, 6, 7） |

### 核心设计

`synthesize_gt_scoring_result(questions)` 生成完美评分，使 GT 和 Belief 共享 `prepare_reconstruction_input_from_scoring()` 的约束提取逻辑。

`corrupt_scoring_result(gt_scoring, questions, η)` 按 noise rate η 翻转 GT 答案，支持 uniform 和 confusion-guided 两种模式。

不修改任何现有文件。
