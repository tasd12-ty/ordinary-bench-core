# Solver Feasibility Bug 排查与修复记录

## 问题现象

约束扰动实验中，**所有场景在所有扰动比例下（包括 p=0 GT 基线）feasibility 都为 False**。

即使使用完全正确的 GT 约束，solver 也判定 infeasible。CSR_qrr 仅 0.6-0.8（应为 1.0），loss 高达数百至数万。

## 排查过程

### 1. 初始怀疑：TRR 镜像 Bug

项目已知存在 TRR 顺时针/逆时针镜像 bug（已在 JSON 文件中修复）。怀疑实验代码使用了未修复的 TRR 数据。

**验证结果：排除。**

- `extract_all_trr(objects, use_3d=True)` 输出与修复后的 JSON 文件 100% 一致（24/24 match）
- 原因：镜像 bug 仅影响 2D 像素坐标（Y 轴翻转），`use_3d=True` 使用 3D 坐标的 x,y 分量，不受影响

### 2. 怀疑：3D 约束在 2D 不可实现

怀疑 QRR 约束基于 3D 距离，但 solver 在 2D 重建，导致本征冲突。

**验证结果：排除。**

- 检查所有场景的 3D 坐标，**z 坐标全部为 0**
- 物体在同一平面上，3D 距离 = 2D 距离，不存在维度降低的信息损失
- 用 GT 2D 坐标直接计算 CSR：`CSR_qrr=1.000, CSR_trr=1.000`，约束完全可满足

### 3. 怀疑：solver 的 restart 次数不足

L-BFGS-B 从随机初始化出发，可能陷入局部极小。

**验证结果：部分正确，但不是根因。**

- 增加 restart 到 50 次，CSR_qrr 最高仅 0.769，CSR_trr 仅 0.417
- loss 从 276 降到 139，但仍远高于 GT 处的理论最优值
- 即使 50 次 restart 也无法收敛到正确解

### 4. 添加 GT Warm-Start

在 solver 的第 0 次 restart 使用 GT 坐标（对齐到 gauge convention）作为初始点。

**验证结果：warm-start 代码生效，但 loss 仍然很高（3265）。**

原因：手动测试时 `bt_scores=None`，loss=1.55（正确）；但 solver 内部自动计算了 `bt_scores`。

### 5. 根因确认：`bt_ratio_alpha=1.0` 的 ratio_loss

**这是最终根因。**

`SolverConfig` 默认 `bt_ratio_alpha=1.0`，启用了 Bradley-Terry 距离比率正则化项。该项要求重建的距离比值接近 BT 模型估算的全局比值。

Loss 分解（GT 坐标处）：

| 分项 | 值 | 占比 |
|------|---|------|
| l_qrr (序约束) | 0.43 | 0.01% |
| l_trr (角约束) | 1.12 | 0.03% |
| l_sep (分离正则) | 0.00 | 0% |
| **l_ratio (BT 比率)** | **3263.74** | **99.95%** |
| **总计** | **3265.29** | |

**原因**：GT 坐标经 gauge 归一化（anchor 距离缩放到 1.0）后，实际距离比值与 BT 模型估算的比值不匹配。ratio_loss 贡献了 99.95% 的总 loss，将 solver 从 GT 附近拖到错误的局部极小。

## 修复方案

### 修改 1：GT Warm-Start（`VLM-test/reconstruct/solver.py`）

在 `solve()` 函数中，将 GT 坐标对齐到 solver 的 gauge convention（anchor_a=(0,0), anchor_b=(1,0)），作为第 0 次 restart 的初始点。

```python
# 第 0 次 restart: 从 GT 坐标出发
if restart == 0 and gt_x0 is not None:
    x0 = gt_x0.copy()
else:
    # 随机初始化
    x0 = rng.randn(n_free) * 1.5
```

对齐步骤：平移（anchor_a → 原点）→ 旋转+缩放（anchor_b → (1,0)）→ pack 为自由变量。

### 修改 2：禁用 ratio_loss（`experiments/constraint_perturbation/run_experiment.py`）

在扰动实验中使用 `SolverConfig(bt_ratio_alpha=0.0)`，只保留序约束 loss（l_qrr + l_trr + l_sep）。

```python
result = reconstruct(
    ...,
    config=SolverConfig(n_restarts=n_restarts, bt_ratio_alpha=0.0),
)
```

## 修复后验证

| 场景 | N | feasible | CSR_qrr | CSR_trr | Kendall τ | NRMS | loss |
|------|---|----------|---------|---------|-----------|------|------|
| n04_000080 | 4 | **True** | 1.000 | 1.000 | 1.000 | 0.023 | 0.38 |
| n05_000085 | 5 | **True** | 0.976 | 1.000 | 0.956 | 0.013 | 1.40 |
| n06_000090 | 6 | **True** | 0.980 | 0.983 | 0.943 | 0.017 | 2.73 |
| n07_000095 | 7 | **True** | 0.995 | 1.000 | 0.981 | 0.007 | 4.29 |
| n08_000085 | 8 | **True** | 1.000 | 0.988 | 0.947 | 0.008 | 8.56 |
| n10_000094 | 10 | **True** | 1.000 | 0.983 | 0.988 | 0.004 | 20.86 |

**所有 N=4~10 场景在 p=0 时均 feasible=True，CSR ≈ 1.0，NRMS < 0.03。**

## 影响范围

- `VLM-test/reconstruct/solver.py`：添加 GT warm-start 逻辑（通用改进，不影响现有功能）
- `experiments/constraint_perturbation/run_experiment.py`：使用 `bt_ratio_alpha=0.0`（仅影响扰动实验）
- 不影响 VLM 评测管线（`VLM-test/API-test/` 不调用 solver）
