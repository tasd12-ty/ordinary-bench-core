# 时空 VLM 论文写作资料表

更新时间：2026-03-12

这份表的目标不是列全所有相关工作，而是把 `ordinary-bench` 当前最适合写成论文的主线，和本地已经下载好的 PDF 对齐。

## 1. 推荐主线

推荐第一版主文聚焦三层：

1. `3D-rendered controlled worlds` 上的静态空间与动态时序评测。
2. 从回答反推出 `scene belief` 的行为反演与场景重建。
3. 用错图/无图、拒答、重述一致性等协议分离视觉贡献和语言先验。

不建议第一版把论文写成“完整 4D 世界模型论文”，因为当前仓库动态部分还没有像静态主链路那样 fully closed。

## 2. 直接支撑主文的问题定义

| 主题 | 论文 | 本地 PDF | 用法 |
| --- | --- | --- | --- |
| 空间知觉 | SpatialVLM | `papers/pdfs/2024-chen-et-al-spatialvlm-cvpr.pdf` | 支撑“VLM 空间推理需要专门评测” |
| 感知与推理脱节 | BLINK | `papers/pdfs/2024-fu-et-al-blink.pdf` | 支撑“模型能看见但不一定真正感知” |
| 视觉评测方法反思 | MMStar | `papers/pdfs/2024-chen-et-al-mmstar.pdf` | 支撑“纯答题基准可能错判模型能力” |
| 语言先验污染 | VLInd-Bench | `papers/pdfs/2025-lee-et-al-vlind-bench.pdf` | 支撑“三条件实验必须存在” |

## 3. 直接支撑时间知觉与视频主实验

| 主题 | 论文 | 本地 PDF | 用法 |
| --- | --- | --- | --- |
| 综合视频评测 | Video-MME | `papers/pdfs/2024-fu-et-al-video-mme.pdf` | 支撑“视频能力不能只看单一任务” |
| 长视频多任务 | MLVU | `papers/pdfs/2024-zhou-et-al-mlvu.pdf` | 支撑长时程、多任务视频理解 |
| 时间推理专门评测 | TempCompass | `papers/pdfs/2024-liu-et-al-tempcompass.pdf` | 支撑顺序、时间一致性、事件层任务 |
| 时间敏感模型 | TimeChat | `papers/pdfs/2024-ren-et-al-timechat-cvpr.pdf` | 支撑“时间建模是额外模块，不会自然出现” |
| 视频时刻理解 | VTimeLLM | `papers/pdfs/2024-huang-et-al-vtimellm-cvpr.pdf` | 支撑 moment-level 时间定位和时间 grounding |
| 4D 世界视角 | VLM4D | `papers/pdfs/2025-zhou-et-al-vlm4d-iccv.pdf` | 支撑“从视频 QA 走向 4D 世界理解”的叙事 |
| 4D 场景表示 | Feature4X | `papers/pdfs/2025-zhou-et-al-feature4x.pdf` | 支撑场景恢复、场景想象、未来扩展 |

## 4. 直接支撑行为反演与可靠性分析

| 主题 | 论文 | 本地 PDF | 用法 |
| --- | --- | --- | --- |
| 拒答比乱答更可靠 | Reliable VQA | `papers/pdfs/2022-whitehead-et-al-reliable-visual-question-answering-abstain-rather-than-answer-incorrectly.pdf` | 支撑 selective QA |
| 选择性回答 | Selectively Answering Visual Questions | `papers/pdfs/2024-eisenschlos-et-al-selectively-answering-visual-questions.pdf` | 支撑允许 `unknown` / `ambiguous` |
| 一致性与不确定性 | Consistency and Uncertainty | `papers/pdfs/2024-khan-et-al-consistency-and-uncertainty-selective-vqa-black-box-vlms.pdf` | 支撑重述一致性与黑盒可靠性分析 |
| 语言模型中的时空结构 | Language Models Represent Space and Time | `papers/pdfs/2023-gurnee-and-tegmark-language-models-represent-space-and-time.pdf` | 支撑“模型内部可能存在时空结构，但主文应走行为检验路线” |

## 5. 直接支撑场景重建与理论部分

| 主题 | 论文 | 本地 PDF | 用法 |
| --- | --- | --- | --- |
| ordinal embedding | Generalized Non-metric MDS | `papers/pdfs/2007-agarwal-generalized-non-metric-multidimensional-scaling.pdf` | 支撑距离序约束到坐标嵌入 |
| 噪声比较鲁棒性 | Robust Ordinal Embedding | `papers/pdfs/2018-ma-xu-cao-robust-ordinal-embedding-from-contaminated-comparisons.pdf` | 支撑 belief reconstruction 不应只看正确答案 |
| 欧氏可实现性 | Euclidean Distance Matrices | `papers/pdfs/2015-dokmanic-parhizkar-ranieri-vetterli-euclidean-distance-matrices.pdf` | 支撑“有序约束不等于几何可实现” |
| 方位唯一性 | Bearing Rigidity | `papers/pdfs/2019-zhao-bearing-rigidity-theory-and-applications.pdf` | 支撑方向约束的对称性与唯一性 |
| 角约束唯一性 | Angle Rigidity | `papers/pdfs/2021-chen-cao-li-angle-rigidity-2d.pdf` | 支撑 TRR 类约束的几何解释 |

## 6. 合成数据基准方法论（新增）

| 主题 | 论文 | 本地 PDF | 用法 |
| --- | --- | --- | --- |
| 合成 VQA 基准 | CLEVR (Johnson et al. 2017) | 需下载 | 最直接类比：受控合成场景评测视觉推理，牺牲生态效度换实验控制 |
| 合成视频推理 | CLEVRER (Yi et al. 2020) | 需下载 | 支撑”受控视频评测”叙事，补充动态模块的 related work |
| VLM 幻觉评测 | POPE (Li et al. 2023) | 需下载 | 支撑”语言先验驱动的伪空间判断”论述 |

## 7. VLM 空间基准补充（新增）

| 主题 | 论文 | 本地 PDF | 用法 |
| --- | --- | --- | --- |
| 视觉空间关系 | VSR (Liu et al. 2023) | 需下载 | Related Work 必须覆盖的空间关系基准 |
| 空间方位理解 | What'sUp (Kamath et al. 2023) | 需下载 | 与 Ordinary-Bench 最接近的空间方位评测 |

## 8. 神经科学方法论参考（新增）

| 主题 | 论文 | 本地 PDF | 用法 |
| --- | --- | --- | --- |
| 表征相似性分析 | RSA (Kriegeskorte et al. 2008) | 需下载 | 支撑”从行为观测反推表征结构”的方法论类比 |
| 深度神经网络与视觉皮层 | Yamins & DiCarlo 2016 | 需下载 | 支撑”比较模型与生物系统的行为”的宏观叙事 |

## 9. 可用于”场景重绘 / 关系图抽取”扩展

| 主题 | 论文 | 本地 PDF | 用法 |
| --- | --- | --- | --- |
| 弱监督 scene graph | LLM4SGG | `papers/pdfs/2024-kim-et-al-llm4sgg-large-language-model-for-weakly-supervised-scene-graph-generation.pdf` | 支撑 direct extraction 路线 |
| 开放世界 scene graph | Open-World Scene Graph Generation | `papers/pdfs/2025-dutta-et-al-open-world-scene-graph-generation-using-vision-language-models.pdf` | 支撑开放词表关系抽取 |

## 10. 推荐在正文中保留的叙事句子

可以直接围绕下面三句写摘要和引言：

1. 当前 VLM 评测过于依赖单题正确率，无法回答“模型究竟看见了什么”。
2. 如果把 VLM 输出转化为时空约束并进行场景级重建，许多看似正确的局部判断会暴露为不一致、欠约束或多模态的伪表征。
3. 在 3D 渲染的受控世界中，错图/无图、乱序/反转、遮挡/缺帧等神经科学式操控实验，可以把视觉贡献、语言先验和时间知觉缺陷分离出来。

## 11. 写作时必须控制的表述

- 不要把当前动态模块写成 `full 3D physical world benchmark`。
- 更准确的表述是：`rendered 3D environments with object-centric 2D/2.5D spatiotemporal annotations`。
- 不要宣称已经解释了模型的内部神经机制。
- 更准确的表述是：`a neuroscience-inspired behavioral reverse-engineering framework`。

## 12. 最小可发版本建议

如果近期要收敛成一篇能发的稿件，建议结构是：

1. 静态空间 benchmark 与多视角结果。
2. three-condition + selective QA + consistency 的视觉贡献分析。
3. scene belief reconstruction 作为核心方法贡献。
4. 动态时间知觉实验作为强补充或第二主实验。
5. 场景重绘 / 补全 / 未来想象作为案例分析，而不是主评测唯一支柱。
