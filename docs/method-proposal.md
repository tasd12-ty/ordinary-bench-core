# 基于 QRR/TRR 的 2D 场景重建方法提案

## 1. 提案目标

本提案的目标很明确：

- 在 `ordinary-bench/VLM-test` 中新增一个 `reconstruct/` 模块
- 输入 VLM 答对的 QRR 和 TRR 约束
- 输出一个 **2D 重建结果**
- 同时判断：
  - 是否存在可行解
  - 是单模态还是多模态
  - 当前重建是“紧”还是“松”

本提案**不做 3D 重建**。  
理由很简单：

- 当前 TRR 只定义在 XY 平面
- 当前场景近似共面
- 直接做 3D 会引入无用自由度和额外歧义

因此，本提案的唯一目标是：

- **做一个可运行、可解释、可评估的 2D 重建 MVP**

---

## 2. 本次实现范围

本次只做以下内容：

1. 从 `scoring.py` 输出中提取正确的 QRR / TRR 约束
2. 在 2D 平面中进行重建
3. 做轻量的符号预检查
4. 用多起点数值优化求可行解
5. 对多次求解结果做聚类
6. 输出代表解和一组基础诊断指标

本次**明确不做**以下内容：

- 3D 重建
- EDM / SDP 松弛
- dReal / interval exact solver
- 完整的“所有可行解”盒覆盖
- 复杂后验采样

这些内容以后可以做，但不进入第一版。

---

## 3. 输入与输出

### 3.1 输入

输入固定为三部分：

1. `qrr_constraints`
   来自答对的 QRR 题

2. `trr_constraints`
   来自答对的 TRR 题

3. `object_ids`
   参与重建的全部物体 id

如果有 GT，则额外输入：

4. `gt_positions`
   仅用于评估，不参与求解

### 3.2 输出

第一版输出固定为下面这个结果结构：

```python
ReconstructResult(
    feasible: bool,
    status: str,
    positions: dict[str, np.ndarray],
    metrics: dict,
    K_geom: int,
    all_solutions: list,
    feasibility_checks: dict,
)
```

字段定义固定如下：

- `feasible`
  是否找到了满足约束的可行解

- `status`
  只能取以下 4 个值之一：
  - `infeasible`
  - `underconstrained`
  - `single_mode`
  - `multimodal`

- `positions`
  最佳代表解，格式为 `obj_id -> [x, y]`

- `metrics`
  至少包含：
  - `csr_qrr`
  - `csr_trr`
  - `spread`
  - `kendall_tau`
  - `nrms`（有 GT 时）

- `K_geom`
  多起点解聚类后的几何模态数

- `all_solutions`
  所有 restart 的求解结果

- `feasibility_checks`
  至少包含：
  - `cycle_free`
  - `trr_consistent`
  - `connected`
  - `n_components`

---

## 4. 方法总览

方法分三步，顺序固定，不做分支设计。

### 第一步：符号预检查

目标：

- 先排除显式矛盾
- 先识别明显欠约束的情况

做法：

1. 处理 QRR
   - `~=` 约束先合并
   - `<` / `>` 建有向图
   - 检查是否有环

2. 处理 TRR
   - 每条 TRR 转成角扇区
   - 相同三元组的扇区求交
   - 若交为空，则直接判为矛盾

3. 处理约束超图
   - 检查是否连通
   - 统计每个物体参与次数

这一步不求坐标，只做判定和清洗。

### 第二步：2D 数值重建

目标：

- 在 2D 中找到满足约束的坐标配置

做法：

1. 先做 gauge fixing
2. 构造联合损失函数
3. 用多起点优化求多个候选解

### 第三步：多解分析

目标：

- 判断是单模态还是多模态
- 判断可行区域是紧还是松

做法：

1. 收集所有 restart 的解
2. 在统一 gauge 下直接比较解之间的距离
3. 做聚类得到 `K_geom`
4. 计算整体 spread 和每个物体的不确定性

---

## 5. 具体方法设计

## 5.1 Gauge Fixing

本提案固定使用 **3-anchor 方案**，不使用其他 gauge。

选择三个 anchor 物体 `a, b, c`，施加以下约束：

- `x_a = (0, 0)`
- `x_b = (1, 0)`
- `y_c >= 0`

这三个条件的作用是：

- 消除平移
- 消除旋转
- 消除缩放
- 消除镜像

anchor 的选择规则也固定：

- 选择在约束中出现次数最多的三个物体
- 如果出现次数并列，优先选同时出现在 QRR 和 TRR 中的物体

这样做的原因很直接：

- 出现次数越多，位置越稳定
- gauge 越稳定，优化越不容易发散

---

## 5.2 QRR 损失函数

QRR 本质是距离比值上的排序，不是原始距离差值排序。  
因此第一版固定使用 **log-domain ranking loss**。

设：

- `d1 = ||x_i - x_j|| + eps`
- `d2 = ||x_k - x_l|| + eps`
- `delta = log(d1) - log(d2)`

则损失定义固定为：

### 对 `<`

```python
L_qrr_lt = softplus(delta + margin)
```

### 对 `>`

```python
L_qrr_gt = softplus(-delta + margin)
```

### 对 `~=`

```python
L_qrr_eq = huber(delta, delta_eq)
```

超参数固定为第一版默认值：

- `margin = 0.1`
- `delta_eq = 0.1`
- `eps = 1e-6`

为什么这样定：

- `log(d)` 天然对缩放不敏感
- `softplus` 平滑，适合 L-BFGS-B
- `huber` 不会因为个别误差过大而把整体拉坏

第一版不引入其他 QRR 损失形式。

---

## 5.3 TRR 损失函数

TRR 的语义不是“逼近某个精确角度”，而是“落在某个允许扇区内”。  
因此第一版固定使用 **扇区容忍损失**。

设：

- `u = normalize(x_ref2 - x_ref1)`
- `v = normalize(x_target - x_ref1)`
- `alpha` 是目标方向中心角
- `tol` 是允许半宽

把 `u` 旋转 `alpha` 得到目标方向向量 `u_alpha`，然后计算：

```python
cos_diff = dot(u_alpha, v)
```

TRR 损失固定定义为：

```python
L_trr = softplus((cos(tol) - cos_diff) / tau_ang)
```

其中：

- hour 约束：`tol = 15°`
- quadrant 约束：`tol = 45°`
- `tau_ang = 0.1`

这个设计的含义非常明确：

- 如果目标点落在扇区内，损失接近 0
- 如果目标点落在扇区外，损失随越界程度增加

第一版不做中心角回归，也不做复杂的角分布建模。

---

## 5.4 分离正则

为了防止多个点塌缩在一起，第一版固定加入分离正则：

```python
L_sep = sum_{i<j} relu(eps_sep - ||x_i - x_j||)
```

默认参数：

- `eps_sep = 0.05`

最终总损失固定为：

```python
L = sum(L_qrr) + sum(L_trr) + lambda_sep * L_sep
```

默认参数：

- `lambda_sep = 1.0`

第一版不再加入其他正则项。

---

## 5.5 求解器

第一版求解器固定如下：

- 优化器：`SciPy L-BFGS-B`
- 初始化：随机高斯初始化
- 重启次数：`n_restarts = 10`

具体流程固定为：

1. 选 anchor
2. 将非 anchor 变量展开成优化向量
3. 随机初始化
4. 调 `scipy.optimize.minimize(..., method="L-BFGS-B")`
5. 收集 10 次 restart 结果
6. 选最优解作为代表解

第一版不使用：

- PyTorch
- Adam
- SGLD
- MCMC

原因很简单：

- `N` 小
- 参数量低
- SciPy 实现成本最低
- 更容易调试和复现

---

## 6. 可行性判定与状态判定

### 6.1 可行性判定

第一版采用统一标准：

- 若符号预检查已发现矛盾，则 `feasible = False`
- 若数值优化后最优解的 `csr_qrr` 和 `csr_trr` 都达到阈值，则 `feasible = True`
- 否则 `feasible = False`

阈值固定为：

- `csr_qrr >= 0.95`
- `csr_trr >= 0.95`

第一版不搞模糊判断，不返回 “maybe feasible”。

### 6.2 状态判定

状态 `status` 的规则固定如下：

1. 如果 `feasible = False`
   - `status = "infeasible"`

2. 如果 `feasible = True` 且 `K_geom = 1` 且 `spread <= 0.10`
   - `status = "single_mode"`

3. 如果 `feasible = True` 且 `K_geom = 1` 且 `spread > 0.10`
   - `status = "underconstrained"`

4. 如果 `feasible = True` 且 `K_geom > 1`
   - `status = "multimodal"`

这里不再引入别的状态名称。

---

## 7. 聚类与 spread

### 7.1 为什么不需要 Procrustes

由于所有解都已经在统一的 3-anchor gauge 下，

- 平移已消除
- 旋转已消除
- 缩放已消除
- 镜像已消除

因此第一版**不做 Procrustes 对齐**，直接比较坐标即可。

### 7.2 聚类规则

第一版固定使用简单阈值聚类：

- 如果两个解的 RMS 距离小于 `0.10`，视为同一模态
- 否则视为不同模态

基于此得到：

- `K_geom`

### 7.3 Spread 定义

第一版把 `spread` 定义为：

- 所有可行解相对于代表解的 RMS 距离平均值

这一定义简单、稳定、可解释。

---

## 8. 评估指标

第一版固定计算以下指标。

### 无 GT 也能算

- `csr_qrr`
  重建结果满足多少比例的 QRR

- `csr_trr`
  重建结果满足多少比例的 TRR

- `K_geom`
  几何模态数

- `spread`
  可行解区域大小

- `per_object_uncertainty`
  每个物体在所有解中的位置方差

### 有 GT 时再算

- `nrms`
  重建结果与 GT 的归一化 RMS 误差

- `kendall_tau`
  重建出的距离排序与 GT 距离排序的一致性

在评价优先级上，第一版固定采用：

1. `csr_qrr` / `csr_trr`
2. `K_geom`
3. `spread`
4. `kendall_tau`
5. `nrms`

这样做的理由是：

- 我们的目标首先是满足序约束
- 不是追求绝对坐标数值最小误差

---

## 9. 模块结构

本提案固定采用以下目录结构：

```text
VLM-test/reconstruct/
├── __init__.py
├── constraints.py
├── solver.py
├── evaluate.py
├── pipeline.py
└── utils.py
```

职责划分固定如下：

- `constraints.py`
  负责 QRR / TRR 约束解析、去重、DAG、扇区和超图检查

- `solver.py`
  负责 gauge fixing、loss、L-BFGS-B 优化、多起点求解

- `evaluate.py`
  负责 CSR、spread、K_geom、Kendall tau、NRMS 计算

- `pipeline.py`
  负责从 scoring 输出直接跑完整流程

- `utils.py`
  放 union-find、角度转换、RMS 距离等通用函数

---

## 10. API 设计

第一版只暴露两个主入口。

### 10.1 直接从约束重建

```python
def reconstruct(
    qrr_constraints: list[dict],
    trr_constraints: list[dict],
    object_ids: list[str],
    gt_positions: dict | None = None,
    n_restarts: int = 10,
) -> ReconstructResult:
    ...
```

### 10.2 从 scoring 结果直接重建

```python
def reconstruct_from_scoring(
    scoring_result: dict,
    questions: list[dict],
    gt_positions: dict | None = None,
    n_restarts: int = 10,
) -> ReconstructResult:
    ...
```

第一版不再提供其他入口。

---

## 11. 验收标准

第一版是否通过，按以下标准判断。

### 功能验收

1. 能从 `scoring.py` 的 `per_question` 中正确提取约束
2. 能完成 QRR 环检测、TRR 扇区一致性检查、超图连通性检查
3. 能输出一个 2D 代表解
4. 能输出 `K_geom` 和 `spread`
5. 能输出 `csr_qrr` 和 `csr_trr`

### 数值验收

在 GT 全正确约束上：

1. `csr_qrr >= 0.99`
2. `csr_trr >= 0.99`
3. `K_geom = 1`
4. `spread <= 0.10`

在随机丢弃 50% 约束后：

1. 仍能稳定求出可行解
2. `csr_qrr >= 0.90`
3. `csr_trr >= 0.90`
4. `spread` 合理增大，不出现完全崩溃

这四条必须全部满足，第一版才算通过。

---

## 12. 不做什么

为了避免目标漂移，本提案明确排除以下工作：

- 不做 3D
- 不做 EDM 显式嵌入层
- 不做 exact solver
- 不做复杂概率模型
- 不做自动调参系统
- 不做可视化前端

第一版只解决一个问题：

- **从正确的 QRR / TRR 约束中，稳定地产出一个可解释的 2D 重建结果**

---

## 13. 我建议的实现顺序

实现顺序固定如下：

1. 先写 `constraints.py`
2. 再写 `solver.py`
3. 再写 `evaluate.py`
4. 最后写 `pipeline.py`

测试顺序固定如下：

1. 先测符号预检查
2. 再测单场景求解
3. 再测多 restart 聚类
4. 最后测 GT 对齐指标

这样推进最稳，不会一开始把问题搞复杂。

---

## 14. 审阅重点

这份提案现在最值得你审的只有 5 件事：

1. 是否同意第一版只做 2D
2. 是否同意使用 3-anchor gauge fixing
3. 是否同意 QRR 用 `log-domain + softplus/huber`
4. 是否同意 TRR 用“扇区容忍”而不是“中心角回归”
5. 是否同意第一版只保留当前这组输出字段和验收标准

如果这 5 点通过，这个方法就已经足够明确，可以直接进入实现。
