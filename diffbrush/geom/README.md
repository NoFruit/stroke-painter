# `diffbrush.geom` 开发文档

> **状态：规划中，未实现。** 本文件先于代码建立，作为后续逐个实现 geom 类的设计依据。
> **范围：** 仅描述 `diffbrush/geom/` 包的规划。math（`utils/bezier.py` 等）与绘制（`render_coverage` 等）**不在本文件改动范围**。

---

## 1. 范式与定位

`geom` 包提供**纯几何生产者**：每个类把对应 utils 数学包成「构造时算一次、缓存到 self、按需取值」的对象。它们是真正绘制行为（`render_coverage` / `cut_endcaps` / scatter）的**上游**--只产参数，不碰绘制。

- **math 不动**：`utils/bezier.py` / `stamp.py` 的纯函数保持原样，geom 类委托调用。
- **绘制不动**：`render_coverage` 等绘制接口的签名/行为一律不改；geom 只把它们要的参数算好喂过去。
- **不接线**：本轮只新建 `geom/` 包，不回改 `core/` 笔刷；独立作为新功能开辟。

---

## 2. 边界与限制（硬约束）

1. **r 解耦**：曲线 geom（`BezierGeom` / `LineGeom`）构造**不碰 r**，只负责 1D 解析式。r 相关的膨胀（tube）/采样密度（d）归 `AABBGeom` / `StampGeom`。
2. **绘制侧零改动**：思考与编辑都不触 `render_coverage` / `cut_endcaps` / scatter。
3. **不回改笔刷 core**：geom 类独立新建，笔刷后续再接入。
4. **采样集 = 带权采样集**：`StampGeom` 的「采样集」含 `centers` 与 `weights`（软计数激活），二者都是 `render_coverage` 的必需输入。

---

## 3. 五个 geom 类

### 3.1 `BezierGeom`（r 无关）

二次 Bézier 1D 解析式缓存。

- **构造**：`BezierGeom(p0, p1, p2)` —— 控制点 (B,2)，**空间无关**（给什么空间，几何量就在什么空间）。
- **缓存**：
  - `.a, .b, .c` —— 系数（`bezier_coefficients`）
  - `.length` —— 弧长 L（`bezier_length` 闭式）
  - `.extrema` —— (B,2) 极值 t（lazy，`bezier_extrema`）
  - `.aabb` —— `(aabb_min, aabb_max)` **曲线** AABB（lazy，候选 {0,1,extrema} 求值取 min/max；**r 无关**）
- **方法**：
  - `.abc` —— property，返回 (a,b,c)
  - `.eval(t)` —— B(t)（`bezier_eval`）
  - `.derivative(t)` —— B'(t)（`bezier_derivative`）
  - `.sample(s_targets, M)` —— **弧长参数化采样**：给目标弧长 (B,K)，返回 centers (B,K,2)。内部 = `arc_length_to_t(a,b,c,s,M)` -> `bezier_eval(a,b,c,t)`。r 无关（只吃 s_targets）；`M` 作为参数传入（构造器保持纯几何）。

### 3.2 `LineGeom`（r 无关）

直线 1D 解析式缓存（与 `BezierGeom` 平行）。

- **构造**：`LineGeom(p0, p2)` —— 端点 (B,2)。
- **缓存**：
  - `.length` —— ‖P2−P0‖（解析）
  - `.direction` —— 单位方向 u（端点切线，供 `CapsGeom` 用）
  - `.aabb` —— 端点 min/max（**曲线** AABB，r 无关；直线无内部极值）
- **方法**：
  - `.sample(s_targets)` —— 直线采样：`t=(s/L).clamp(max=1)`，`centers=(1−t)P0+t·P2`。无 `M`（直线弧长反演是闭式 s/L）。

### 3.3 `StampGeom`（几何无关）

绘制前夕的几何无关数据 = 采样集 + 半径。

- **构造**：`StampGeom(centers, weights, r_px)` —— 持有 `render_coverage` 要的量。
  - `centers` (B,K,2) —— 采样点（由曲线 geom 的 `.sample` 产出，`StampGeom` 不知其来源）
  - `weights` (B,K) —— 软计数激活（采样集的一部分）
  - `r_px` (B,1) —— 像素半径
- **字段**：`.centers, .weights, .r_px`。
- **不持有** `grid_coords` / `disk_softness`（那些是 `render_coverage` 的绘制侧输入，调用时传）。
- **通用 stamp 参数设置**（`d` / `s_targets` / `weights`，由 L+r 产出）归 `StampGeom` 的 classmethod 还是 `stamp.py` helper —— **待定**（实现 `StampGeom` 时定，§8）。

### 3.4 `AABBGeom`（吃 r）

曲线 AABB + r -> 渲染框派生链，**各级独立缓存**。

- **构造**：`AABBGeom(curve_aabb_min, curve_aabb_max, r_px, aabb_pad)` —— 吃曲线 AABB（来自 `BezierGeom` / `LineGeom` 的 `.aabb`）+ r。
- **缓存**（派生链，全部保留，互不覆写）：
  - `.tube` —— 曲线 AABB ± r（tx0,ty0,tx1,ty1 / cx,cy / ex,ey）
  - `.square` —— 取大轴 `emax` 的正方形（half = emax）
  - `.padded` —— `half·(1+aabb_pad)`（pad 作为开关/量保留）
  - `.integer` —— 整数像素框 `aabb_px`(=ceil(2·half))、`ix0`、`iy0`（贴回原点）

### 3.5 `CapsGeom`（r 无关，仅 square）

方头切割的几何量 u0/u2（端点切线）。**只有 square 笔刷实例化**（取代 `None`-跳过范式）。

- **构造**：`CapsGeom(u0, u2)` —— 直接传切线；或工厂 `from_bezier(bg)`（u0=B'(0), u2=B'(1) 归一化）/ `from_line(lg)`（u0=u2=direction）。工厂是否提供 **待定**（§8）。
- **字段**：`.u0, .u2`。
- r 无关（纯切线）。`sharpness` 是绘制侧超参，调用时传。

---

## 4. 关键设计决策（已确认）

| 决策 | 结论 |
| --- | --- |
| 两种 AABB | 曲线 AABB（r 无关，曲线 geom 出）+ 渲染 AABB（`AABBGeom` 吃曲线 AABB+r 出） |
| `AABBGeom` 多级缓存 | tube->square->padded->integer 派生链，各级独立字段，pad 作开关 |
| `StampGeom` 几何无关 | 只持 `{centers, weights, r_px}`，不知曲线类型 |
| 采样集含 weights | 是（`render_coverage` 必需） |
| 采样归属曲线 geom | `.sample(s_targets)` 在 `BezierGeom` / `LineGeom`（曲线专属、r 无关） |
| 采样粒度 | (A) 打包成单方法 `.sample(s, M)` |
| `M` 放置 | `.sample` 参数（构造器保持纯几何）；若恒为 256 可后续提为类默认 |
| `CapsGeom` 独立 | square 专用类，uniform 不实例化（取代 `None`） |

---

## 5. 采样 / 绘制流（笔刷接入后预期形状）

```
bg = BezierGeom(P0p, P1p, P2p)                       # r 无关
# 通用 stamp 参数设置（d/s_targets/weights，由 L+r 产出；归属待定）
d, s_targets, weights, r_px = stamp_setup(bg.length, r, rho, K, alpha, ref)
centers = bg.sample(s_targets, M)                    # 曲线专属、r 无关
sg  = StampGeom(centers, weights, r_px)              # 几何无关持有体
# 绘制侧（本包不碰）：render_coverage(sg.centers, sg.r_px, sg.weights, grid, soft)
```

---

## 6. 与现有代码的对应

| 现有（`core/bezier_square_brush.py` 等） | 归属 |
| --- | --- |
| `bezier_coefficients` + `bezier_length`（abc + L） | `BezierGeom` |
| `bezier_derivative`(0/1) -> u0/u2 | `CapsGeom.from_bezier` |
| `bezier_extrema` + 候选求值 -> 曲线 AABB | `BezierGeom.aabb` |
| `arc_length_to_t` + `bezier_eval` -> centers | `BezierGeom.sample` |
| `soft_stamp_count` + d/s_targets -> weights | 通用 stamp_setup（归属待定） |
| tube/square/pad/整数框 | `AABBGeom` |
| `render_coverage` / `cut_endcaps` / scatter | **绘制侧，不动** |

---

## 7. 实现路线

逐个建，每建一个停一次确认：

1. `bezier_geom.py` —— `BezierGeom`（第一块，统一三处 `abc+L+derivative+extrema+aabb+sample` 重复：`BezierSquare.forward` / `forward_fast` / `BezierUniform.forward`）
2. `line_geom.py` —— `LineGeom`
3. `stamp_geom.py` —— `StampGeom`（届时定 `stamp_setup` 归属）
4. `aabb_geom.py` —— `AABBGeom`
5. `caps_geom.py` —— `CapsGeom`
6. 最后再讨论笔刷 `core` 接入。

---

## 8. 待定项

- `stamp_setup`（d / s_targets / weights 通用设置）归 `StampGeom` classmethod 还是 `stamp.py` helper。
- `M`（arc_length_grid）是否提为 `BezierGeom` 类默认。
- `CapsGeom` 构造用 `from_bezier` / `from_line` 工厂还是直接传 (u0,u2)。
