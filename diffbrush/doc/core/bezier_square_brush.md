# BezierSquareBrush — 单色均匀粗细贝塞尔方刷（OBB-SDF）

**代码路径建议：** `implementations/bezier_square_brush.py`

## 定位

一种单色、粗细均匀的三点 quadratic Bézier 笔刷，**stamp 形状为方形**（非圆形）。
继承自 `BrushBase`。

与 [BezierUniformBrush](bezier_uniform_brush.md) 的差异**仅在 per-pixel SDF 的取形**：
后者每 stamp 是 soft-disk（圆形），本笔刷每 stamp 是 soft-OBB——以 stamp 点为中心、
以 `r` 为两轴半长（边长 `2r` 的正方形）、一轴朝向曲线切线 `u_k` 的定向包围盒，
每像素取到该 OBB 的有符号距离渲染 soft-box，呈方形笔触。曲线、stamp 采样、
Sigmoid 软计数、概率并集、alpha 合成、参数集与 forward 接口**完全一致**。

## 参数集

所有参数均为**归一化输入**（详见 [设计范式](../design_paradigm.md#参数归一化输入)）。
参数维度与布局与 BezierUniformBrush **逐位相同**（`PARAM_DIM = 11`）：

```
Θ = { P₀, P₁, P₂, c, r, α }
```

| 参数 | 含义 | 形状 |
|---|---|---|
| `P₀` | 起点坐标 | `(2,)` |
| `P₁` | 控制点坐标 | `(2,)` |
| `P₂` | 终点坐标 | `(2,)` |
| `c` | 单色 RGB 颜色 | `(3,)` |
| `r` | **方框半轴长**（两轴相同，边长 `2r`） | `(1,)` |
| `α` | 透明度 | `(1,)` |

> **`r` 的含义变化、维度与接口不变。** 与 BezierUniformBrush 同一参数槽
> （位置 9、维度 1），归一化 → 像素 `r·min(H,W)` 也相同；唯一差异是 BezierUniformBrush
> 把它当 disk 半径用，本笔刷把它当方框半轴长用。对调用方而言 `r` 仍是同一无约束参数。

## 数学形式

### 曲线

与 [BezierUniformBrush — 曲线](bezier_uniform_brush.md#曲线) 完全一致：三点 quadratic Bézier
`B(t) = (1-t)²P₀ + 2(1-t)t P₁ + t² P₂`，弧长 `L = ∫‖B'(t)‖dt`（数值积分）。

### 渲染（soft-OBB）

single-color 下 nearest-stamp 取色退化为"像素是否落入笔刷覆盖区"。把 BezierUniformBrush
的 hard disk 替换为**有符号距离到定向方框的 soft 形式**。每个 stamp k：

- 中心 `x_k = B(t_k)`（由弧长 → t 反演得到，与 BezierUniformBrush 同）
- 单位切线 `u_k = B'(t_k) / ‖B'(t_k)‖`，垂直方向 `v_k = (-u_{k,y}, u_{k,x})`
- 半轴 `r`（两轴相同 → 正方形）

像素 `x` 到该 OBB 的有符号距离（标准 2D box SDF，两轴半长均为 `r`）：

```
la = (x − x_k)·u_k            lb = (x − x_k)·v_k          # 局部坐标
qx = |la| − r                 qy = |lb| − r
outside = sqrt( max(qx,0)² + max(qy,0)² + ε )              # ε=1e-12 数值保护
inside  = clamp( max(qx,qy), max=0 )
sdf     = outside + inside                                  # <0 内，=0 边界，>0 外
```

soft-box（相对半轴归一化，与 soft-disk 同构）：

```
box_k(x) = σ( −γ · sdf / r )
```

`sdf = 0`（半轴 `r` 的方框边界）处 → 0.5，内部 → 1，外部 → 0。该形式与 soft-disk
`disk_k = σ(γ·(1 − ‖x−x_k‖/r)) = σ(−γ·sdf_disk/r)`（`sdf_disk = ‖x−x_k‖ − r`）
同构：相同的 `γ/r` 过渡宽度、相同的"0.5 落在特征尺度边界"语义。故 `r` 在两刷中
扮演相同的"可见特征尺度"角色——disk 为半径，方刷为半轴长。

**Stamp 激活权重**（与 BezierUniformBrush 同）：

```
w_k = σ( α_cnt · (L/d − k) )
```

**覆盖并集（soft OR）**：

```
m_k(x) = w_k · box_k(x)
M(x)   = 1 − ∏_{k=1}^{K} ( 1 − m_k(x) )      ∈ [0,1]
```

**Alpha compositing（预乘 RGBA，与 BezierUniformBrush 同）：**

```
A(x) = α · M(x)
RGB(x) = c · A(x)
patch = concat(RGB, A) ∈ (B, 4, H, W)
```

`+1e-12` 数值保护说明：方框内部是**正面积区**上 `outside` 恒为 0（不像 disk 仅在中心
测度零点处为 0），无 eps 则 `sqrt(0)` 反传为 `inf`/`NaN`。eps 与 `bezier.py` 弧长反演的
`denom = s1 - s0 + 1e-12` 同属模块数值保护约定（非参数约束）。

### 退化与限知

- 退化曲线（`B'(t)≡0`，如 `P0=P1=P2`）：`u_k` 未定义（零向量），但 `soft_stamp_count`
  给出的 `w_k≈0` 屏蔽该 stamp，与 BezierUniformBrush 处理退化曲线的机制一致，无新失败模式。
- 二次曲线尖点（`B'(t)=0` 于某 `t`，quadratic 当 `(P1−P0)` 与 `(P2−P1)` 反平行时可能出现）：
  若某 stamp 恰落尖点，该单 stamp 渲染为一个朝向任意的完整方框。测度零情形，非阻塞，
  不做防御处理（[设计范式](../design_paradigm.md#防御性编程)禁防御代码）。

---

## Stamp 采样策略

与 BezierUniformBrush 完全一致：

- 比例参数 ρ，间隔距离 d = r × ρ
- Sigmoid 软计数计算 N：`N = Σ σ(α · (L/d - k))`
- 尾部余段为正常行为，不需处理

详见 [BezierUniformBrush — Stamp 采样策略](bezier_uniform_brush.md#stamp-采样策略)。
