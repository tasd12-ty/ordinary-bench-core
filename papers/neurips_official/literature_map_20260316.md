# Ordinary-Bench 文献地图与阅读记录

日期：2026-03-16

## 检索与阅读流程

本轮按照“搜索 -> 下载 -> 阅读 -> 记录 -> 再搜索”的顺序进行了三轮整理。

第一轮重点补齐近两年视觉 benchmark 与 world model 主线：
- 通用多模态评测：MMMU, SEED-Bench, HallusionBench。
- 空间 / 3D / 多视角 / 遮挡 / 动态：MetaVQA, MotionBench, 3DSRBench, CAPTURE, NuPlanQA。
- world model：Drive-WM, Navigation World Models, MaskGWM, I2-World, HERMES。
- 对象级表征：CObL。

第二轮补齐 ACL / ICLR / NeurIPS 以及 visual CoT / tool-use：
- benchmark / 评测方法：SPHERE, MMMU-Pro, SpatialMQA, COMFORT, WM-ABench。
- 世界模型评测：Evaluating the World Model Implicit in a Generative Model。
- 视觉 CoT / 工具使用 / test-time scaling：Visual Sketchpad, ViperGPT, Visual Program Distillation, VisuoThink, Chain of Multi-modal Thought inference-time scaling, LLaVA-Plus。

第三轮补脉络锚点：
- 心理测量学视角：Defining and Evaluating Basic Spatial Abilities。
- 更干净的物理空间关系识别：Can Transformers Capture Spatial Relations Between Objects?
- 经典世界模型源头：World Models。

选择标准：
- 以 2024-2025 年 CCF-A 会议 / 期刊为主。
- 只在叙事骨架需要时补少量更早的奠基论文。
- 只保留对 Ordinary-Bench 的论证真正有帮助的论文，不为了“多”而堆无关引用。

## 历史脉络梳理

### 1. 从“大而全多模态评测”走向“诊断型评测”

早期的大型多模态 benchmark 往往回答的是“模型大体会不会做多模态题”，例如 MMMU、SEED-Bench。这类 benchmark 的价值在于建立广覆盖能力版图，但它们通常无法分离：
- 视觉证据是否真的必要；
- 模型是否依赖文本先验或选项偏置；
- 局部答对是否对应一致的内部世界表征。

随后出现的诊断型 benchmark，如 MMMU-Pro、HallusionBench、MMStar、VLInd-Bench、BLINK，开始把“评测错配”“语言污染”“视觉不可或缺性”“感知而非知识回忆”单独拿出来测。这一转向与 Ordinary-Bench 最接近，因为我们的目标也不是再造一个“更难的题库”，而是改变评测终点：从题目正确率转向场景信念的可重建性。

### 2. 从二维关系题走向三维、多视角、遮挡与动态

空间 benchmark 的第二条主线是：从简单的左右上下判断，逐步走向更贴近真实空间认知的问题。

这条线大致经历了：
- viewer-centric 的二维关系识别；
- 更干净、物理定义更明确的空间关系分类；
- 带 frame of reference 歧义的空间语言理解；
- 3D 关系、多视角、ego-centric 驾驶场景；
- 遮挡、模式延拓、amodal completion；
- 时间维度上的运动与状态变化。

Ordinary-Bench 站在这条线的下一个位置：我们不只问“能否识别关系”，而是问“关系回答集合是否足以反推出一个统一场景”。这一步把 benchmark 与重建理论接了起来。

### 3. 从“构建 world model”走向“评估 world model 是否真的存在”

world model 文献大多聚焦于学习潜在动力学，用于控制、预测和生成。典型目标是：
- 预测未来帧；
- 预测 3D / 4D occupancy 或 BEV 演化；
- 在潜空间中做规划、仿真或视频生成。

近期也开始出现“如何评估模型到底有没有学到 coherent world model”的工作，如 WM-ABench 与 Vafa 等人的工作。这一方向对 Ordinary-Bench 非常关键：它说明“看起来会答题”“看起来会预测”并不等于真正有一致的世界模型；评价内部世界表征需要更细粒度、更结构化的诊断。

我们的区别非常明确：
- 我们不是训练一个 pixel-space 或 occupancy-space 的生成 world model。
- 我们不要求模型生成未来视频，也不要求它输出 3D 网格或 occupancy。
- 我们评估的是：模型的外显回答，是否定义了一个对象级、代码级、矢量化、与具体尺度数值解耦的场景结构。

### 4. 从心理物理学与 psychometrics 取回“能力分解”和“行为反演”视角

COMFORT 和 Basic Spatial Abilities 这两类工作非常重要，因为它们不再把空间能力视为一个单一分数，而是拆成多个基本能力：
- spatial perception
- spatial relation
- spatial orientation
- mental rotation
- spatial visualization
- frame of reference consistency

这与 Ordinary-Bench 的核心思想高度一致：我们不是直接读取模型参数里的“世界模型”，而是通过密集、受控、可组合的行为 probe 去反演其外显空间信念。也就是说，我们的方法在风格上更接近心理物理学和系统神经科学，而不是普通 QA benchmark。

### 5. 从视觉 CoT / tool-use 提升题目得分，到重新界定“我们到底想测什么”

Visual Sketchpad、ViperGPT、Visual Program Distillation、VisuoThink、LLaVA-Plus 以及 inference-time scaling for multimodal thought 代表了另一条强势路线：通过外部工具、绘图、程序执行、树搜索、长链推理，在 test time 增强模型的解题能力。

这类方法对工程上“把题做对”很有价值，但它们和 Ordinary-Bench 想测的能力不完全一致：
- 它们可以借助额外检测器、OCR、分割器、代码执行器和搜索策略来弥补原生视觉不足。
- 它们优化的是“解题流程”，不一定是“形成稳定的视觉世界表征”。
- 它们可能把任务转化为可计算问题，而不是要求模型本身拥有可重建的场景信念。

因此，这类工作在 related work 中应被明确纳入，但作为“能力增强路线 / 目标错位对照线”，而不是与 Ordinary-Bench 处于同一评测目标上。

## 逐篇记录

### A. 通用多模态 benchmark 与诊断型评测

#### MMMU
- 会议：CVPR 2024
- 核心内容：11.5K 题，覆盖 30 个学科与多种异质图像类型，强调“大学水平”的多模态理解和推理。
- 价值：建立通用 multimodal reasoning 的广覆盖评测版图。
- 局限：更像总体能力普查，不直接测试视觉是否不可或缺，也不测试回答之间的全局一致性。
- 对 Ordinary-Bench 的意义：可作为“现有大 benchmark 主要评测题目层面能力”的代表工作。

#### SEED-Bench
- 会议：CVPR 2024
- 核心内容：分层组织多模态能力，覆盖图像、视频、文本图像交错输入等多类任务。
- 价值：提供能力分层与较大覆盖面。
- 局限：仍以任务正确率为中心，不触及“内部场景是否一致可还原”。
- 对 Ordinary-Bench 的意义：可放在“广谱能力 benchmark”一类，与我们的结构化空间 probe 形成对照。

#### MMMU-Pro
- 会议：ACL 2025
- 核心内容：在 MMMU 基础上去除可由 text-only 解出的题、扩展选项、加入 vision-only 设置。
- 关键发现：很多看似强的结果来自 benchmark shortcut；CoT 有帮助，OCR prompt 帮助有限。
- 价值：证明“视觉必要性”需要被显式控制。
- 对 Ordinary-Bench 的意义：直接支撑我们强调“不能把答对题目等同于形成视觉世界表征”。

#### HallusionBench
- 会议：CVPR 2024
- 核心内容：构造带控制组的图像-上下文诊断题，分析语言 hallucination 与 visual illusion。
- 关键发现：即使强模型在 question-pair accuracy 上也很低，暴露强烈语言偏置与多种失败模式。
- 对 Ordinary-Bench 的意义：说明诊断型 benchmark 需要控制设计，而不仅是题量；也说明局部正确率不足以证明真实视觉理解。

#### BLINK
- 会议：ECCV 2024
- 核心内容：把 14 类经典感知任务改写为 VLM benchmark，强调“人类眨眼间可做出”的基础视觉感知。
- 关键发现：最强模型也只略高于随机，尤其在相对深度、多图对应、多视角任务上表现差。
- 对 Ordinary-Bench 的意义：与我们的发现高度同向，尤其支持“多视角不一定帮忙，反而暴露模型没有统一场景表征”。

#### MMStar
- 会议：NeurIPS 2024
- 核心内容：指出 benchmark 中存在 visual-unnecessary 和 data leakage 两大问题，并提出更严格的评测标准。
- 价值：把“评测本身是否在测目标能力”变成研究问题。
- 对 Ordinary-Bench 的意义：我们的 related work 应明确承接这条线，把 Ordinary-Bench 定位成“进一步把评价终点改成 scene reconstructability”。

#### VLInd-Bench
- 会议：NAACL Findings 2025
- 核心内容：显式测量语言先验如何污染视觉回答。
- 对 Ordinary-Bench 的意义：可作为“语言污染”线的重要引用，和 HallusionBench / MMStar 共同支撑评测错配段落。

### B. 空间、三维、多视角、遮挡与动态 benchmark

#### SpatialMQA
- 会议：ACL 2025
- 核心内容：以人工标注方式构建更干净的 objective-world spatial relation benchmark，避免过度依赖 bbox 或纯先验作答。
- 关键发现：SOTA MLLM 准确率只有 48.14%，远低于人类 98.40%。
- 对 Ordinary-Bench 的意义：可作为“即便在更干净的空间关系评测上，模型仍远弱于人类”的直接证据。

#### SPHERE
- 会议：ACL 2025
- 核心内容：分层测空间能力，从基础技能到多技能整合再到高级推理。
- 关键发现：distance/proximity、egocentric vs allocentric、spatial logic 都是明显盲点。
- 对 Ordinary-Bench 的意义：提供“层级化空间能力”这一路径，但仍停留在 item-level；我们进一步要求这些回答可共同支撑同一场景。

#### COMFORT
- 会议：ICLR 2025
- 核心内容：考察 frame of reference 歧义、多语言与跨文化空间表达。
- 关键发现：模型缺乏鲁棒性、一致性和 frame flexibility，英语习惯压制其他语言惯例。
- 对 Ordinary-Bench 的意义：非常适合放进 introduction / related work 的 cognitive angle 中，说明空间理解不只是“左右前后识别”，还涉及坐标系选择与一致性。

#### Defining and Evaluating Basic Spatial Abilities
- 会议：ACL 2025
- 核心内容：从 psychometrics 出发，把 VLM 空间能力拆成五类基本空间能力。
- 关键发现：模型在不同基本能力上差异很大，小模型有时胜过大模型；ToT 优于简单 CoT，但仍受结构限制。
- 对 Ordinary-Bench 的意义：是我们“psychophysics / psychometrics 风格评测”叙事的关键支点。

#### Can Transformers Capture Spatial Relations Between Objects?
- 会议：ICLR 2024
- 核心内容：强调空间关系应使用物理上清晰定义，而非语言学上含混定义；提出更干净的数据与 transformer 方法。
- 关键发现：现有方法在 physically grounded spatial relation 上表现并不理想。
- 对 Ordinary-Bench 的意义：说明“关系定义是否清晰”是 benchmark 成败的前提，也支持我们使用严格 QRR / TRR 关系语义。

#### 3DSRBench
- 会议：ICCV 2025
- 核心内容：系统评估自然图像中的 3D spatial reasoning，覆盖高度、位置、朝向、多对象关系，并引入 FlipEval。
- 关键发现：模型在非常多的 3D awareness 维度上脆弱，且在非常规视角下明显退化。
- 对 Ordinary-Bench 的意义：表明视角变化和三维关系确实是现有模型的薄弱点；我们的多视角场景 probe 与之形成强连接。

#### CAPTURE
- 会议：ICCV 2025
- 核心内容：通过遮挡下的模式延拓与 amodal counting 评估空间推理和 world modeling。
- 关键发现：模型在遮挡条件下明显退化，给出辅助位置信息后性能上升，说明错误部分来自无法“想象”被遮挡结构。
- 对 Ordinary-Bench 的意义：支持“如果模型真的有场景信念，就应能在部分缺失证据下维持一致结构”这一论点。

#### MetaVQA
- 会议：CVPR 2025
- 核心内容：面向 embodied scene understanding，联合 VQA 与 closed-loop simulation，强调 object-centric driving scenes。
- 价值：把 spatial awareness 和 embodied understanding 结合起来。
- 局限：最终仍以 agent / driving 成功为目标，不以场景可重建性为评测终点。
- 对 Ordinary-Bench 的意义：适合放在 embodied benchmark 一段，与我们“读取模型脑海中的场景”动机形成联系。

#### NuPlanQA
- 会议：ICCV 2025
- 核心内容：大规模 multi-view driving VQA 数据与评测，强调多视角、BEV 与 ego-centric reasoning。
- 价值：多视角驾驶场景是 Ordinary-Bench 未来动态扩展的重要邻域。
- 局限：仍主要是 driving-specific QA 与多视角场景理解，不直接测试回答集合的几何一致性。

#### MotionBench
- 会议：CVPR 2025
- 核心内容：评估细粒度 motion-level video understanding，而非 story-level 总结。
- 关键发现：现有视频 VLM 在最基础的 motion-level 感知上依然薄弱。
- 对 Ordinary-Bench 的意义：为我们扩展到动态场景提供支撑。时间维度上的失败往往先出现在最局部的运动关系上。

### C. world model 与 world-model 评测

#### World Models
- 年份：2018
- 核心内容：用生成模型学习环境的紧凑时空表征，并在“梦境”中训练控制器。
- 历史意义：world model 路线的经典源头之一。
- 对 Ordinary-Bench 的意义：可在 related work 中作为世界模型概念源头的简短引用。

#### Drive-WM
- 会议：CVPR 2024
- 核心内容：多视角 driving world model，用于未来视频生成与规划。
- 目标：基于动作想象多种未来，按 image-based reward 选轨迹。
- 对 Ordinary-Bench 的区别：Drive-WM 追求未来视频生成与规划；我们不追求视频生成，而是测试 VLM 是否已有对象级、可反演的场景信念。

#### Navigation World Models
- 会议：CVPR 2025
- 核心内容：条件扩散 transformer 预测未来观测，并用于导航轨迹评估与规划。
- 对 Ordinary-Bench 的区别：它的“world model”是 controllable video generation and planning；我们的“world model”更接近外显 scene belief 的可重建性诊断。

#### MaskGWM
- 会议：CVPR 2025
- 核心内容：通过视频 mask reconstruction 强化 driving world model 的泛化与长时预测。
- 对 Ordinary-Bench 的区别：仍是生成 / 预测导向，而不是关系约束导向。

#### I2-World
- 会议：ICCV 2025
- 核心内容：通过 intra-scene / inter-scene tokenization 做高效 4D occupancy forecasting。
- 对 Ordinary-Bench 的区别：它输出 occupancy 级 4D scene forecast；我们输出的是与尺度数值解耦的对象关系结构。

#### HERMES
- 会议：ICCV 2025
- 核心内容：把 3D scene understanding 与 scene generation 放进统一 driving world model。
- 价值：说明“understanding + generation”正在被统一建模。
- 对 Ordinary-Bench 的区别：HERMES 仍位于驾驶仿真与 3D 生成空间；我们评估的是回答所定义的结构世界，而不是生成器本身。

#### WM-ABench
- 会议：ACL Findings 2025
- 核心内容：把 world model 拆成 perception 与 prediction 两阶段，做原子化评估。
- 关键发现：许多最新 VLM 在最基础的 motion / future-state 推断上接近随机。
- 对 Ordinary-Bench 的意义：与我们的结论高度同向，也支持“atomic evaluation 比总体分数更能揭露真实短板”。

#### Evaluating the World Model Implicit in a Generative Model
- 会议：NeurIPS 2024
- 核心内容：提出更严格的评价指标，指出许多 generative model 看似学到 world model，实则内部状态并不连贯。
- 关键思想：单步预测或表面诊断可能严重高估 coherent world model。
- 对 Ordinary-Bench 的意义：这篇论文是我们“为什么要用结构一致性而不是单题准确率”论证的强支撑。

### D. visual CoT / tool-use / inference-time scaling

#### ViperGPT
- 会议：ICCV 2023
- 核心内容：让代码生成模型把视觉模块组合成 Python 程序来完成复杂视觉查询。
- 价值：显式、可解释、可调试地把“视觉处理”和“逻辑/计算”拆开。
- 对 Ordinary-Bench 的意义：说明提高分数可以依赖工具链组合，而不必来自原生视觉世界表征。

#### LLaVA-Plus
- 年份：2023
- 核心内容：多模态 agent 能调用技能库中的工具，执行视觉理解、生成、知识检索及组合任务。
- 对 Ordinary-Bench 的意义：可作为“工具增强多模态代理”代表，与纯 VLM 内部表征能力区分。

#### Visual Program Distillation
- 年份：2024
- 核心内容：先用 LLM + 工具生成正确程序，再把 reasoning distill 进单个 VLM。
- 关键价值：说明工具链既可以直接推理，也可以蒸馏成模型能力。
- 对 Ordinary-Bench 的意义：非常适合在 related work 中写成“工具使用并不等同于真实视觉能力，但可作为提升答题表现的工程路线”。

#### Visual Sketchpad
- 会议：NeurIPS 2024
- 核心内容：给模型一个可画线、框、标记的视觉草图板，让模型在视觉空间中做 CoT。
- 关键发现：能显著提升几何、地图、空间和复杂视觉任务表现。
- 对 Ordinary-Bench 的意义：它提升的是解题过程中的中间操作能力；这与我们评测“模型原本脑海中的场景是否一致”不是同一个目标。

#### VisuoThink
- 会议：ACL 2025
- 核心内容：多模态 tree search，把视觉提示与语言推理交织进 slow thinking。
- 对 Ordinary-Bench 的意义：代表 test-time search 方向。可在 related work 中说明：更长推理链可能帮助答题，但这不自动说明原生空间表征更好。

#### Investigating Inference-time Scaling for Chain of Multi-modal Thought
- 会议：ACL Findings 2025
- 核心内容：系统比较 text-only thought 与 multi-modal thought 在 inference-time scaling 下的表现。
- 关键发现：多模态 thought 确实有帮助，但 token 开销更高，也表明 test-time compute 是一个重要变量。
- 对 Ordinary-Bench 的意义：可作为实验设计中的对照线，未来可将“是否允许外部思考与工具”做成评测设置。

### E. 对象级 / amodal / 向量化场景表示

#### CObL
- 会议：ICCV 2025
- 核心内容：无用户提示地推断按遮挡顺序排列的 object layers，实现多物体 amodal completion。
- 价值：非常接近“对象级场景表示”的视觉传统。
- 对 Ordinary-Bench 的区别：CObL 输出的是图层式像素 / 物体层表示；我们追求的是对象关系图式的、代码级、矢量化重建。

## 对写作的直接启示

### 1. abstract / introduction 的主线应该这样组织

第一段不是从 benchmark 开始，而是从视觉的根本目的开始：
- 视觉系统的目标不是回答单题，而是形成关于世界的可行动、可预测、可重构的内部表征。
- 人类视觉在很大程度上可以被理解为：从有限视角证据出发，建立对象及其关系组成的结构世界。
- 如果足够准确的偏序关系可以近似重建场景，那么对 VLM 做密集关系问答，本质上是在读取其外显场景信念。

随后切入 Ordinary-Bench：
- 现有 benchmark 多停留在题目级得分。
- 现有 world model 多停留在生成、预测、控制。
- 现有 visual CoT / tool-use 多停留在解题增强。
- 我们关心的是更基础的问题：VLM 是否已经拥有一个可由回答集合反演出来的、稳定的对象级空间世界。

### 2. related work 最好拆成五段

建议结构：
- 心理物理学 / psychometrics / 行为反演。
- 通用多模态 benchmark 与评测错配。
- 空间 / 3D / 多视角 / 遮挡 / 动态 benchmark。
- world model 与 world-model evaluation。
- visual CoT / tool-use / inference-time scaling。

### 3. world model 段落必须明确写出的差异

建议写法：
- world model 关注 latent dynamics, future observation forecasting, occupancy / BEV / video generation, planning。
- Ordinary-Bench 不训练 world model，不要求生成未来帧，也不要求输出数值几何。
- 我们测试的是 VLM 回答能否外化为对象级、关系级、与尺度数值解耦的向量化场景表示。
- 因此 Ordinary-Bench 更像是对“外显 scene belief”的结构诊断，而不是对“生成式世界模拟器”的建模。

### 4. visual CoT / tool-use 段落必须明确写出的差异

建议写法：
- 这些方法可能显著提升视觉题目的最终正确率。
- 但它们通过外部工具、程序执行、树搜索、额外绘图空间和更长推理链来弥补原生视觉不足。
- 因而它们优化的是“做题机制”，不一定是在测或提升“已有的稳定视觉表征”。
- Ordinary-Bench 的价值在于：即便允许模型作答，我们仍可追问这些回答能否共同支持一个真实、一致、可还原的场景。

## 对实验设计的启示

### 1. 静态实验不应只报告单题准确率

至少应包含：
- 单图 QRR / TRR。
- 多视角 QRR / TRR。
- 从回答集合恢复场景的可实现率、冲突率、重建误差。
- scene-level consistency 指标，而非只看 item accuracy。

### 2. 动态实验应成为论文的重要扩展，而不是附录小点

可以设计四类动态实验：
- 时序 QRR：对象间远近偏序随时间变化。
- 时序 TRR：相对方向关系随时间变化。
- change-point probing：问模型何时发生关系翻转、接近、交叉、遮挡。
- trajectory-consistency probing：分别从单帧、短序列、多视角序列提问，比较其隐含重建是否相容。

### 3. 可以设置“原生能力”与“增强能力”两套 protocol

建议对照：
- 原生 VLM 直接回答。
- 加 CoT。
- 加 visual CoT / sketch / tool-use。
- 加外部几何工具或程序搜索。

这样可以更清楚地区分：
- 模型本身是否拥有稳定场景表征；
- 模型是否只是通过额外计算流程把题做对。
