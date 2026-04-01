# ORDINARY-BENCH 完整管线总览

## 一、全局管线概览

```mermaid
graph TB
    GEN[Blender CLEVR 渲染]
    SCENE[scene JSON]
    IMG1[单视角图片]
    IMG4[多视角图片 x4]
    QGEN[问题生成 v2]
    QRR[QRR 距离比较]
    TRR[TRR 钟面方向]
    FDR[FDR 全距离排序]
    VLMEVAL[VLM 评估]
    SCORING[评分]
    CONFLICT[冲突检测与解决]
    RECON[空间重建]
    ANALYSIS[分析与可视化]

    GEN --> SCENE
    GEN --> IMG1
    GEN --> IMG4
    SCENE --> QGEN
    QGEN --> QRR
    QGEN --> TRR
    QGEN --> FDR
    QRR --> VLMEVAL
    TRR --> VLMEVAL
    FDR --> VLMEVAL
    IMG1 --> VLMEVAL
    IMG4 --> VLMEVAL
    VLMEVAL --> SCORING
    SCORING --> CONFLICT
    SCORING --> RECON
    CONFLICT --> RECON
    RECON --> ANALYSIS
    SCORING --> ANALYSIS
```

---

## 二、Blender CLEVR 数据生成

```mermaid
graph TB
    CFG[config.toml + CLI]
    GEN[generate.py]
    PIPE[pipeline.py 调度 Blender]
    BLENDER[render_multiview.py]
    ASSETS[base_scene + shapes + materials]

    SCENE_OUT[scenes/scene_id.json]
    SV_OUT[images/single_view/scene_id.png]
    MV_OUT[images/multi_view/scene_id/view_0-3.png]
    TOP_OUT[images/top_view/scene_id.png]

    CFG --> GEN
    GEN --> PIPE
    PIPE --> BLENDER
    ASSETS --> BLENDER
    BLENDER --> SCENE_OUT
    BLENDER --> SV_OUT
    BLENDER --> MV_OUT
    BLENDER --> TOP_OUT
```

**配置优先级**: 默认值 < config.toml < --preset < CLI 参数

**Splits**: n04 / n05 / n06 / n07 / n08 / n09 / n10 (每个 split 固定物体数)

---

## 三、问题生成

```mermaid
graph TB
    SCENE[scene JSON N 个物体]
    PARSE[parse_objects 提取坐标和属性]

    QD[QRR disjoint: 3 x C N 4]
    QS[QRR shared_anchor: N x C N-1 2]
    FILT[边界过滤 tau=0.10]
    TP[TRR: P N 3 有序三元组]
    CLK[atan2 映射 1-12 小时]
    FA[FDR: N 个锚点排序]
    TIE[tie_groups 并列处理]

    BATCH[make_batches 分批]
    OUT[questions/qrr trr fdr/scene_id.json]

    SCENE --> PARSE
    PARSE --> QD
    PARSE --> QS
    QD --> FILT
    QS --> FILT
    PARSE --> TP
    TP --> CLK
    PARSE --> FA
    FA --> TIE
    FILT --> BATCH
    CLK --> BATCH
    TIE --> BATCH
    BATCH --> OUT
```

**N=10 规模**: QRR 约 990 题 + TRR 720 题 + FDR 10 题 = 约 1720 题/场景

---

## 四、VLM 评估

```mermaid
graph TB
    JOB[JobSpec .toml 配置]
    DISC[发现场景 按 split 过滤]
    LOADQ[加载问题 JSON]
    LOADIMG[加载图片 单视角或多视角]
    GROUP[按题型分组 再按 batch 分批]
    PROMPT[构造 user prompt]
    SYS[选择 system prompt]
    CALL[调用 VLM API]
    PARSE[parse_batch_response]
    CHECK{缺答率大于 20%}
    CORRECT[ReAct 补问]
    MERGE[合并回答]
    SCORE[逐题评分 + 场景聚合]
    SAVE[保存结果 JSON]
    AGG[跨场景汇总]

    JOB --> DISC
    DISC --> LOADQ
    DISC --> LOADIMG
    LOADQ --> GROUP
    GROUP --> PROMPT
    LOADIMG --> PROMPT
    SYS --> CALL
    PROMPT --> CALL
    CALL --> PARSE
    PARSE --> CHECK
    CHECK -- 是 --> CORRECT
    CORRECT --> CALL
    CHECK -- 否 --> MERGE
    MERGE --> SCORE
    SCORE --> SAVE
    SAVE --> AGG
```

**Provider 支持**: OpenAI / OpenRouter / Gemini / DashScope / Mock

---

## 五、评分体系

```mermaid
graph LR
    subgraph QRR
        Q1[精确匹配 comparator]
        Q2[按 disjoint shared_anchor 分别统计]
    end

    subgraph TRR
        T1[hour 精确匹配]
        T2[quadrant 同象限]
        T3[adjacent 正负1小时]
    end

    subgraph FDR
        F1[exact 完全匹配]
        F2[kendall tau]
        F3[pairwise 成对正确率]
        F4[top-1 最近物体]
    end
```

---

## 六、冲突检测与解决

```mermaid
graph TB
    INPUT[VLM Belief 模式回答]
    PREP[提取所有 QRR 约束 含 FDR 分解]
    DAG[构建距离偏序 DAG]
    FAS[FAS 最小反馈弧集]
    MAP[映射回 question_id]
    REASK[对冲突题重问 VLM]
    CONV[收敛检测]
    CLASS[分类: 噪声 vs 系统性]
    OUTPUT[修正后约束集]

    INPUT --> PREP
    PREP --> DAG
    DAG --> FAS
    FAS --> MAP
    MAP --> REASK
    REASK --> CONV
    CONV --> CLASS
    CLASS --> OUTPUT
```

---

## 七、空间重建

```mermaid
graph TB
    GT[GT 模式: 仅正确约束]
    BEL[Belief 模式: 所有 VLM 回答]

    CMODES{约束组合}
    CALL[all]
    CQRR[qrr_only]
    CTRR[trr_only]
    CFDR[fdr_only]

    POSET[DistancePoset 等价类合并]
    SECTOR[Angular Sectors TRR 角度范围]
    CYCLE{环路检测}
    FAIL[infeasible]

    ANCHOR[锚定 3 物体消除自由度]
    LOSS[hinge loss + 角度 loss + 正则化]
    OPT[L-BFGS-B 10 次多重启]
    CLUST[解聚类]

    CSR[CSR 约束满足率]
    KT[Kendall tau]
    KG[K_geom 模态数]
    NRL[NRL + p_value]

    GT --> CMODES
    BEL --> CMODES
    CMODES --> CALL
    CMODES --> CQRR
    CMODES --> CTRR
    CMODES --> CFDR
    CALL --> POSET
    CALL --> SECTOR
    CQRR --> POSET
    CTRR --> SECTOR
    CFDR --> POSET
    POSET --> CYCLE
    SECTOR --> CYCLE
    CYCLE -- 可行 --> ANCHOR
    CYCLE -- 不可行 --> FAIL
    ANCHOR --> LOSS
    LOSS --> OPT
    OPT --> CLUST
    CLUST --> CSR
    CLUST --> KT
    CLUST --> KG
    CLUST --> NRL
```

---

## 八、分析与输出

```mermaid
graph TB
    RES[逐场景评分 JSON]
    REC[重建结果]

    T1[准确率汇总 按模型 x 题型 x split]
    T2[重建质量 CSR tau K_geom]
    T4[一致性 传递性 互易性]
    SVG[SVG 2D 场景图]
    CHART[柱状图 按 n_objects 分组]
    XLS[Excel 报告]

    RES --> T1
    RES --> T4
    REC --> T2
    REC --> SVG
    T1 --> CHART
    T2 --> CHART
    T1 --> XLS
    T2 --> XLS
    T4 --> XLS
```

---

## 九、子集图片消融实验

> 核心问题: VLM 是否受图中无关物体干扰? 能否识别不存在的物体并拒答?

```mermaid
graph TB
    ORIG[父场景 N=6..10 物体]
    ENUM[enumerate_subsets.py<br>枚举 C N 4 子集]
    MANIFEST[manifest.json]
    RENDER[render_subsets.py<br>Blender 重渲染 只留 4 物体]
    SUBIMG[912 张子集图片]
    MASTER[generate_master_questions.py<br>全量 QRR bank<br>disjoint + shared_anchor + FDR 分解]
    ASSIGN[assign_subset_questions.py<br>全量问题分配<br>answerable + N/A 拒答]
    SUBQ[每子集全量问题<br>含 answerable 标记]
    EVAL[run_subset_eval.py<br>VLM 评估 含 N/A 解析]
    SCORE[评分: 准确率 + 拒答率 + 幻觉率]
    ANALYSIS[analyze_results.py<br>full vs subset 对比]

    ORIG --> ENUM
    ENUM --> MANIFEST
    MANIFEST --> RENDER
    RENDER --> SUBIMG
    ORIG --> MASTER
    MASTER --> ASSIGN
    MANIFEST --> ASSIGN
    ASSIGN --> SUBQ
    SUBIMG --> EVAL
    SUBQ --> EVAL
    EVAL --> SCORE
    SCORE --> ANALYSIS
```

---

## 十、端到端执行路径

```mermaid
graph LR
    A[1 Blender 渲染]
    B[2 问题生成]
    C[3 VLM 评估]
    D[4 冲突检测]
    E[5 重建]
    F[6 分析]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
```

| 阶段 | 入口脚本 | 输入 | 输出 |
|------|---------|------|------|
| 1 渲染 | data-gen/generate.py | config.toml | scenes/ images/ |
| 2 问题 | VLM-test/generate_questions_v2.py | scenes/ | questions/qrr,trr,fdr/ |
| 3 评估 | VLM-test/API-test/run_eval.py | questions/ images/ | results/ |
| 4 冲突 | VLM-test/run_conflict_resolution.py | results/ | 修正约束 |
| 5 重建 | VLM-test/reconstruct/pipeline.py | 约束集 | positions, metrics |
| 6 分析 | VLM-test/analysis/run_analysis.py | results/ metrics/ | Excel, SVG |

---

## 附: 关键目录结构

```
ordinary-bench-core/
├── data-gen/                     # CLEVR Blender 渲染后端
│   ├── generate.py               #   入口
│   ├── pipeline.py               #   Blender 子进程调度
│   └── blender/                  #   Blender 脚本和资产
├── datasets/test-data/           # 140 场景测试集
│   ├── scenes/                   #   场景 JSON
│   ├── images/single_view/       #   单视角图片
│   └── questions/{qrr,trr,fdr}/  #   按题型分目录
├── VLM-test/
│   ├── dsl/                      # 空间推理 DSL
│   ├── question_bank.py          # 问题枚举
│   ├── extraction.py             # 场景解析
│   ├── generate_questions_v2.py  # 问题生成入口
│   ├── API-test/
│   │   ├── run_eval.py           # 评估入口
│   │   ├── eval_engine.py        # 批量调用 + ReAct
│   │   ├── providers/            # VLM 适配器
│   │   ├── scoring.py            # 评分
│   │   └── jobs/*.toml           # 任务配置
│   ├── reconstruct/
│   │   ├── pipeline.py           # 重建入口
│   │   ├── constraints.py        # DAG + 扇区
│   │   ├── solver.py             # L-BFGS-B
│   │   └── evaluate.py           # CSR tau K_geom
│   ├── conflict_resolution/      # 冲突检测与解决
│   └── analysis/                 # 分析与可视化
└── docs/                         # 文档
```
