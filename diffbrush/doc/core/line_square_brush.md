# LineSquareBrush — 单色均匀粗细直线方刷（OBB-SDF）

**代码路径：** `implementations/line_square_brush.py`

## 定位

一种单色、粗细均匀的直线笔刷，**stamp 形状为方形**（非圆形）。
继承自 `BrushBase`。

与 [UniformLineBrush](uniform_line_brush.md) 的差异**仅在 per-pixel SDF 的取形**：
后者每 stamp 是 soft-disk（圆形），本笔刷每 stamp 是 soft-OBB——以 stamp 点为中心、
以 `r` 为两轴半长（边长 `2r` 的正方形）、一轴朝向直线方向 `u` 的定向包围盒，
每像素取到该 OBB 的有符号距离渲染 soft-box，呈方形笔触。直线、stamp 采样、
Sigmoid 软计数、概率并集、alpha 合成、参数集与 forward 接口**完全一致**。

## 参数集

所有参数均为**归一化输入**（详见 [设计范式](../design_paradigm.md#参数归一化输入)）。
参数维度与布局与 UniformLineBrush **逐位相同**（`PARAM_DIM = 9`）：

```
Θ = { P₀, P₂, c, r, α }
```

| 参数 | 含义 | 形状 |
|---|---|---|
| `P₀` | 起点坐标 | `(2,)` |
| `P₂` | 终点坐标 | `(2,)` |
| `c` | 单色 RGB 颜色 | `(3,)` |
| `r` | **方框半轴长**（两轴相同，边长 `2r`） | `(1,)` |
| `α` | 透明度 | `(1,)` |

> **`r` 的含义变化、维度与接口不变。** 与 UniformLineBrush 同一参数槽
> （位置 7、维度 1），归一化 → 像素 `r·min(H,W)` 也相同；唯一差异是 UniformLineBrush
> 把它当 disk 半径用，本笔刷把它当方框半轴长用。对调用方而言 `r` 仍是同一无约束参数。

## 数学形式

### 直线

与 [UniformLineBrush — 直线](uniform_line_brush.md#直线) 完全一致：

```
B(t) = (1−t)·P₀ + t·P₂,  t ∈ [0,1]
L = ||P₂ − P₀||₂
```

弧长可解析得到，t = s/L，无需数值反演。

### 渲染（soft-OBB）

single-color 下 nearest-stamp 取色退化为"像素是否落入笔刷覆盖区"。把 UniformLineBrush
的 hard disk 替换为**有符号距离到定向方框的 soft 形式**。每个 stamp k：

- 中心 `x_k = B(t_k)`（由弧长 → t 直接计算，t_k = s_k/L）
- 单位切线 `u` = `(P₂ − P₀) / ||P₂ − P₀||`（**常量**，所有 stamp 共享同一朝向）
- 垂直方向 `v = (−u_y, u_x)`
- 半轴 `r`（两轴相同 → 正方形）

像素 `x` 到该 OBB 的有符号距离（标准 2D box SDF，两轴半长均为 `r`）：

```
la = (x − x_k)·u            lb = (x − x_k)·v          # 局部坐标
qx = |la| − r               qy = |lb| − r
outside = sqrt( max(qx,0)² + max(qy,0)² + ε )          # ε=1e-12 数值保护
inside  = clamp( max(qx,qy), max=0 )
sdf     = outside + inside                              # <0 内，=0 边界，>0 外
```

soft-box（相对半轴归一化，与 soft-disk 同构）：

```
box_k(x) = σ( −γ · sdf / r )
```

`sdf = 0`（半轴 `r` 的方框边界）处 → 0.5，内部 → 1，外部 → 0。该形式与 soft-disk
`disk_k = σ(γ·(1 − ‖x−x_k‖/r)) = σ(−γ·sdf_disk/r)`（`sdf_disk = ‖x−x_k‖ − r`）
同构：相同的 `γ/r` 过渡宽度、相同的"0.5 落在特征尺度边界"语义。故 `r` 在两刷中
扮演相同的"可见特征尺度"角色——disk 为半径，方刷为半轴长。

**Stamp 激活权重**（与 UniformLineBrush 同）：

```
w_k = σ( α_cnt · (L/d − k) )
```

**覆盖并集（soft OR）**：

```
m_k(x) = w_k · box_k(x)
M(x)   = 1 − ∏_{k=1}^{K} ( 1 − m_k(x) )      ∈ [0,1]
```

**Alpha compositing（预乘 RGBA，与 UniformLineBrush 同）：**

```
A(x) = α · M(x)
RGB(x) = c · A(x)
patch = concat(RGB, A) ∈ (B, 4, H, W)
```

`+1e-12` 数值保护说明：方框内部是**正面积区**上 `outside` 恒为 0（不像 disk 仅在中心
测度零点处为 0），无 eps 则 `sqrt(0)` 反传为 `inf`/`NaN`。

### 退化与限知

- 退化直线（`P0 = P2`，`L = 0`）：`t_k` 通过 `+1e-12` 保护分母，`t_k` 恒为 1（终点），
  所有 stamp 中心重合于同一点。切线 `u` 未定义（零向量除以 1e-12），但 `soft_stamp_count`
  给出的 `w_k ≈ 0` 屏蔽这些 stamp（`L=0` → `L/d=0` → 所有 `w_k ≈ 0`），实际不渲染。
  与 UniformLineBrush 处理退化直线的机制一致，无新失败模式。

---

## Stamp 采样策略

与 UniformLineBrush 完全一致：

- 比例参数 ρ，间隔距离 d = r × ρ
- Sigmoid 软计数计算 N：`N = Σ σ(α · (L/d - k))`
- 尾部余段为正常行为，不需处理

详见 [UniformLineBrush — Stamp 采样策略](uniform_line_brush.md#stamp-采样策略)。
