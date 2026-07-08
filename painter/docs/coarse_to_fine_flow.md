# coarse_to_fine 数据流与接口规定（现状基线）

> **文档定位**：coarse-to-fine 流程的数据流转与接口契约 ground truth。**「数据精简为
> 切片+置换」重构已落地**（`Level` 精简为 `image + transform(TileTransform)`，参数维度无关），
> 本文档反映重构后现状。第 6 节耦合清单标注本次已处理的点（#1/#2/#5）与仍待解耦的点；
> 「规范化 + 参数无关化」总目标的其余项见 §8。
>
> **范围**：coarse-to-fine 主干（`CoarseToFine` 类 + `main` 三层循环）及其直接上下游
> （`image_input` / `error_map` / `loss_space` / `main` 内联渲染函数）。
> **不在范围**：neube 风格化内部、笔刷库 / loss 库内部实现（只看它们对主干暴露的接口）。
>
> **事实来源**：本文档据 `ot-brush-optimize-workspace` 当前代码（`main.py` / `coarse_to_fine.py` /
> `error_map.py` / `loss_space.py` / `image_input.py` / `device.py` / `core_imports.py`）。
> 代码是 ground truth；如代码改了本文档需同步。

---

## 0. 一句话总览

```
target (1,4,H,W) ──► CoarseToFine ──► pyramid: List[Level]   （每级 target 切片 + 仿射）
                                              │
              ┌───────────────────────────────┘  canvas_full 跨级累积
              ▼
   for level in pyramid:            （粗→细）
     canvas_batch = slice(canvas_full)[level]      （把累积画布切到本级分辨率）
     for stroke-batch in _N_STROKES:               （每级画几批，B=grid_n² 一片一笔）
       em.compute(canvas_batch, target_batch)      （误差引导 + top 块矩特征）
       raw = init_raw(em, b)                       （矩引导初始参数，无约束）
       for epoch in _EPOCHS_PER_STROKE:            （render→loss→backward→step）
         params  = reparam(raw)                    （raw → 合法域 params）
         comp,A = _forward_strokes(params, canvas_batch, brush, patch)   （一片一笔 over 合成）
         loss    = loss_space.forward(comp, target_batch, A)
       params = reparam(raw).detach()              （固化）
       canvas_batch = _forward_strokes(params, ...)      （tile-res 更新，作下批底子）
       params_full  = c2f.params_to_full(params, level)   （切片归一 → 原图归一）
       canvas_full  = _commit_strokes(params_full, canvas_full, brush)  （full-res 正式光栅化）
   save_output(canvas_full)
```

**两条分辨率线**：优化在 **tile-res**（切片 patch，如 128/64）上做，commit 在 **full-res**
（原图 H×W）上落。`params_to_full` 是两线之间的翻译桥。

---

## 1. 顶层组件与全局状态

main 启动时把下列全局量加载到位（范式：不做防御性编程，任一为 None 下游自然崩）：

| 组件 | 来源 | 角色 | None 情形 |
|---|---|---|---|
| `dev.device` | `device.py` | 全局 device（CUDA→MPS→CPU） | 永不为 None |
| `img_input.target` | `image_input.img_input.load()` | 目标图 `(1,4,H,W)` 直通 RGBA `[0,1]` | 文件缺失→None |
| `img_input.canvas` | 同上 | 画布 `(1,4,H,W)`（缺失则全 0 空白，尺寸取自 target） | target 也缺→None |
| `ci.brush` / `ci.loss` | `core_imports.py` | 两边模块命名空间 | 导入失败→None |
| `ci.BezierUniformBrush` | 同上 | 笔刷类符号 | 失败→None |
| `brush` | `ci.BezierUniformBrush()` 实例 | 可微笔刷实例（`forward(params, patch_size)→(B,4,ph,pw)`） | — |
| `loss_space` | `loss_space.py` 单例 | 组合损失环境（l1+ot+grad+area，单一 cache owner） | — |
| `c2f` | `CoarseToFine(target)` | 切片金字塔（优化空间类） | target None→构造抛 |
| `em` | `ErrorMap(target)` | 误差信息类（算引导图 + top 块矩特征，不进 loss） | target None→构造抛 |
| `applicator` | `ci.BrushStyleApplicator(dev.device)` | neube 风格化器（可选，仅前向备用） | 构造失败→None，循环不依赖 |

**main 内联的优化环境**（非独立模块，全在 `main.py` 文件作用域）：
`reparam` / `init_raw` / `_forward_strokes` / `_commit_strokes` / `_commit_strokes_styled` +
硬编码配置 `_N_STROKES` / `_EPOCHS_PER_STROKE` / `_OPT_LR` / `_COMMIT_CHUNK` +
重参数化约束表 `_PARAM_CONSTRAINTS` / `_PARAM_TRANSFORMS`。

---

## 2. CoarseToFine 类：输入输出接口

### 2.1 构造

```python
CoarseToFine(target, n_target=16, factor=2, cap=128)   # target: (1,4,H,W) RGBA float [0,1]
```

构造即完成预处理：自动生成 levels → 对 target 逐级切片 + 降采样 + 算每片坐标置换
（`TileTransform`）→ `self.pyramid: List[Level]`。target 与 canvas 共用同一套切法（`_slice`），保证几何一致。

**配置（构造参数，带默认值=现状值，直接可见、向后兼容）**：

| 参数 | 默认 | 语义 |
|---|---|---|
| `n_target` | 16 | 目标最细网格边长（n×n 的 n） |
| `factor` | 2 | 网格逐级倍率（图像金字塔标准 factor-2） |
| `cap` | 128 | 单切片像素边长上限；`region > cap` 才降采样到 cap |

**levels 自动生成**（`_compute_levels`）：规约到最大的 `factor^k ≤ n_target` 且**同时整除 H 与 W**；
`levels = [1, 2, 4, …, factor^k]`。整除保证同级所有切片尺寸一致 → 可堆成批张量 `(B,4,ph,pw)`。
对 1024×1024 target：`levels=[1,2,4,8,16]`。

**实例属性**：`target (1,4,H,W)`、`H, W`、`levels: List[int]`、`pyramid: List[Level]`、
`n_target / factor / cap`。

### 2.2 Level 数据结构（dataclass，参数维度无关）

数据二元论——`Level` 只管两份数据（**切片** `image` + **置换** `transform`），不感知笔刷
参数是几维、哪几通道是点 / 哪个是 r：

```python
@dataclass
class TileTransform:                 # 坐标置换组（纯数据，参数维度无关）
    point_affine: torch.Tensor       # (B,2,3)  2D 仿射 [[a,b,tx],[c,d,ty]]：切片归一→原图归一
    r_scale: torch.Tensor            # (B,)     半径标量缩放（= a = 切片→全图像素尺度比）

@dataclass
class Level:
    grid_n: int                      # 网格边长（n×n 的 n）
    image: torch.Tensor              # (B,4,ph,pw)  切片像素 RGBA float [0,1]，B=grid_n²，detach（进 optimizer）
    transform: TileTransform         # 坐标置换：point_affine + r_scale（切片↔原图 2D 坐标互转）
    # property: n_tiles  = image.shape[0]                       = grid_n²
    # property: patch_hw = (image.shape[-2], image.shape[-1])   同级一致（批张量化前提）
```

砍掉的旧字段：`row` / `col` / `region_full` / `downsampled` / `affine_full`
（`affine_full` → `transform.point_affine`）。

**device**：`_slice` 里 affine 用 `torch.tensor(..., device=img.device)` 创建 → `transform`
全程在 target device（旧 CPU 瑕疵已修，`params_to_full` 的 `.to` 成无开销对齐）。

### 2.3 对外方法

| 方法 | 签名 | 作用 |
|---|---|---|
| `slice(img)` | `(1,4,H,W) → List[Level]` | 用与 target **完全相同的切法**切任意同尺寸图（canvas 用）。要求 img 与 target 同 (H,W)，否则抛。几何与 `self.pyramid` 一致，仅 image 像素不同。 |
| `params_to_full(params, level)` | `(B,11), Level → (B,11)` | **11 维布局已知时的便利方法**（Level 数据结构参数无关，二者不冲突）。内部读 `level.transform.point_affine` 把切片归一参数翻译到原图归一参数（commit/画回全图用）。**只动几何维**：P0/P1/P2 仿射、r 乘 `a`(=`transform.r_scale`) 缩放保像素半径、c/α 不变。可微（不 detach）。 |
| `iter_levels()` / `__iter__` / `__len__` | — | 逐级迭代（粗→细）。 |
| `memory_bytes()` | `→ int` | 金字塔所有切片张量字节数（常驻开销，各级可重复访问）。 |
| `summary_str()` | `→ str` | 诊断打印（数值，不读图）。 |

私有：`_compute_levels()`、`_slice(img)`（target 与 canvas 共用切法）。

---

## 3. 数据流转全链路（main 三层循环）

### 3.0 预处理（main 启动）

```
img_input.load()                           # target / canvas 到位
target = img_input.target                  # (1,4,H,W) 直通 RGBA
brush  = ci.BezierUniformBrush()
applicator = ci.BrushStyleApplicator(...)  # 可选，失败→None
c2f = CoarseToFine(target)                 # 建金字塔（target 切片 + 仿射）
em  = ErrorMap(target)                     # 持 target 全图作源
canvas_full = img_input.canvas.clone()     # 正式画布（full-res 累积载体，单一 canvas）
```

### 3.1 level 循环（粗→细，`for li, lvl in enumerate(c2f.pyramid)`）

每级取监督量与本级 canvas 初值：

```
grid_n       = lvl.grid_n
b            = lvl.n_tiles                # = grid_n²
patch_size   = lvl.patch_hw                # 本级切片分辨率（tile-res）
target_batch = lvl.image                   # (B,4,ph,pw) 监督量 RGBA（detach）
canvas_batch = c2f.slice(canvas_full)[li].image   # 把累积 canvas_full 切到本级 B 片
```

**关键**：`canvas_full` 跨级累积，作为下一级初值；每级开头把它切到本级分辨率得 `canvas_batch`。

### 3.2 stroke-batch 循环（`for si in range(_N_STROKES)`，每级 _N_STROKES 批）

每批 `B=grid_n²` 笔（一片一笔，批 forward 一对一）：

```
em.compute(canvas_batch, target_batch)     # 算误差引导图 + top 块矩特征，缓存到 em
raw = init_raw(em, b, dev.device)          # 矩引导初始 raw (b,11) 无约束，requires_grad
optimizer = torch.optim.RMSprop([raw], lr=_OPT_LR)
```

### 3.3 epoch 循环（`for ei in range(_EPOCHS_PER_STROKE)`）

```
optimizer.zero_grad()
params     = reparam(raw)                                       # raw → 合法域 params (B,11)
composited, A = _forward_strokes(params, canvas_batch, brush, patch_size)   # 一片一笔 over 合成
loss       = loss_space.forward(composited, target_batch, A)    # 组合标量（可 backward）
loss.backward()
optimizer.step()
```

### 3.4 固化与 commit（每批 epoch 完成后）

```
params = reparam(raw).detach()                                  # 固化合法参数

# (1) tile-res 更新 canvas_batch（未翻译 params，与优化器同分辨率，作下批底子）
with torch.no_grad():
    composited, _ = _forward_strokes(params, canvas_batch, brush, patch_size)
    canvas_batch = composited.detach()

# (2) commit：翻译到原图空间 → 串行光栅化到正式画布
params_full  = c2f.params_to_full(params, lvl)                  # 切片归一 → 原图归一 (B,11)
canvas_full  = _commit_strokes(params_full, canvas_full, brush) # full-res 正式光栅化（无梯度）

plt.show()   # 一次
```

**commit 二选一**（单 canvas 注入，main.py:443-444 注释切换）：
- 实色（默认）：`_commit_strokes(params_full, canvas_full, brush)`
- 风格化（手动切）：`_commit_strokes_styled(params_full, canvas_full, brush, applicator)`
  —— `applicator=None` 时自动退化实色。切到 styled 后，风格化笔画 over 进同一个
  `canvas_full`，下一级 `slice` 读到风格化画 = 下批优化底子。

### 3.5 结束

```
out_path = image_input.ImageInput.save_output(canvas_full)   # 落盘 output/output.png（预乘→直通）
```

---

## 4. 接口规定清单（签名 + 张量契约）

> 统一约定：计算空间**全程四维 `(B,4,H,W)` 预乘 RGBA float [0,1]**（B 维 = batch）。
> 几何/切片/降采样/仿射只作用于空间 `(H,W)`，与通道数无关（但全链路写死 4 通道）。

### 4.1 CoarseToFine（coarse_to_fine.py）

| 接口 | 输入 | 输出 | 梯度 | device |
|---|---|---|---|---|
| `CoarseToFine(target, n_target=16, factor=2, cap=128)` | `target (1,4,H,W)` RGBA | `self`（pyramid 就绪） | target detach | target 的 device；transform 同 device |
| `slice(img)` | `img (1,4,H,W)`，须同 (H,W) | `List[Level]`（几何同 pyramid，image 不同） | detach | img 的 device；transform 同 device |
| `params_to_full(params, level)` | `params (B,11)` 切片归一；`level: Level` | `(B,11)` 原图归一 | 可微（纯几何） | 对齐 params.device（transform 已在 target device，.to 常无开销） |

### 4.2 ErrorMap（error_map.py，信息类，不进 loss/backward）

| 接口 | 输入 | 输出 | 备注 |
|---|---|---|---|
| `ErrorMap(target)` | `target (1,4,H,W)` | `self` | 缓存 target 全图作源 |
| `compute(canvas_batch, target_batch)` | 均 `(B,4,ph,pw)` 同分辨率切片 | `self`（链式） | 填充 `abs_diff/grad_map/diff_direct/labels/top_*` |

`compute` 后 `em` 暴露的 top 块矩特征（笔刷无关中间表示，供 `init_raw` 消费）：
`top_centroid (B,2)` / `top_orientation (B,)` / `top_axis_major (B,)` / `top_axis_minor (B,)` /
`top_color (B,3)` / `top_label (B,)` / `top_area (B,)` / `top_score (B,)` / `labels (B,1,ph,pw)`。

### 4.3 main 内联函数（main.py，非模块）

| 函数 | 签名 | 输入契约 | 输出契约 |
|---|---|---|---|
| `reparam(raw)` | `(B,11) → (B,11)` | 无约束 raw | 物理量通道入合法域（c/α∈(0,1), r>0），几何 [0:6] 恒等透传。可微 |
| `init_raw(em, b, device)` | `ErrorMap, int, device → (b,11)` | em 的 top_* 矩特征 | 无约束 raw，`requires_grad=True` |
| `_forward_strokes(params, canvas_batch, brush, patch_size)` | `(b,11),(b,4,ph,pw),Brush,(ph,pw)` | params 带梯度 | `(composited (b,4,ph,pw), A (b,1,ph,pw))` 预乘 over 合成 |
| `_commit_strokes(params_full, canvas_full, brush)` | `(B,11),(1,4,H,W),Brush` | params_full 已翻译 detach | `canvas_full (1,4,H,W)` 累积（detach） |
| `_commit_strokes_styled(params_full, canvas_full, brush, applicator)` | 同上 + applicator | applicator None→退化实色 | 同上 |
| `loss_space.forward(pred_canvas, target=None, brush_alpha=None)` | `pred (B,4,H,W)` 带梯度；`target (B,4,H,W)`；`brush_alpha (B,1,H,W)` | — | 组合标量 loss（可 backward） |

### 4.4 笔刷参数布局（11 维，与 BezierUniformBrush 强耦合）

```
[P0x, P0y, P1x, P1y, P2x, P2y, cR, cG, cB, r, α]
  0    1    2    3    4    5   6   7   8  9  10
```

- 几何 `[0:6]`：P0/P1/P2 三点，**切片归一 [0,1]²** 空间，无 reparam 约束（real 自由漂移）。
- 颜色 `[6:9]`：c RGB ∈ (0,1)（`unit_interval`，logit 双射）。
- 半径 `[9]`：r > 0（`positive`，log 双射）。
- 透明度 `[10]`：α ∈ (0,1)（`unit_interval`，logit 双射）。

`brush.forward(params, patch_size) → (B,4,ph,pw)` 预乘 RGBA。

### 4.5 image_input（image_input.py）

| 接口 | 作用 |
|---|---|
| `img_input.load()` | 加载 target / canvas 到 self（均 `(1,4,H,W)` RGBA） |
| `img_input.target` / `img_input.canvas` | 全局单例张量 |
| `ImageInput.save_output(canvas)` | 预乘→直通 RGBA，落盘 `output/output.png` |

---

## 5. 空间与坐标系约定

### 5.1 三套坐标系

| 坐标系 | 范围 | 用途 |
|---|---|---|
| **切片归一 [0,1]²** | 笔刷参数 P0/P1/P2 语义所在 | 优化在切片 patch 上做，参数语义为"本切片内归一坐标" |
| **原图归一 [0,1]²** | 全图归一 | commit / 正式光栅化（`brush.forward(params_full, full_size)`） |
| **原图像素 [0,H)×[0,W)** | 像素索引 | 切片几何（y0/x0/step、切片像素边长 region） |

### 5.2 transform.point_affine 语义（切片归一 → 原图归一）

`Level.transform.point_affine (B,2,3)`，形式 `[[a,b,tx],[c,d,ty]]`，`x_full = a·x + b·y + tx`。

等分网格下：`a = region_norm/W`、`d = region_norm/H`、`b = c = 0`、`tx = x0/W`、`ty = y0/H`
（`region_norm = (step_r+step_c)/2`，H=W 时即 step）；`transform.r_scale = a`（= 切片→全图像素尺度比）。

### 5.3 params_to_full 的几何变换（读 transform.point_affine；r 用 a = transform.r_scale）

```
P0/P1/P2 (切片归一) → 原图归一：x_full = a·x + tx ,  y_full = d·y + ty
r (切片归一)        → 原图归一：r_full = r·a        （a = transform.r_scale = region_norm/W = 切片→全图像素尺度比，
                                                       保像素半径一致：细级 a 小→细笔，粗级 a 大→粗笔）
c (颜色) / α        → 不变
```

### 5.4 降采样规则

切片原图精度 `region × region`（`region` = 切片原图像素边长，等分网格下 = step）：
- `region > cap` → `F.adaptive_avg_pool2d(crop, (cap,cap))`，patch=(cap,cap)。
- `region ≤ cap` → 原精度（不放大、不超采样）。

1024×1024 / cap=128 实例：grid=1/2/4 降采样到 128；grid=8 原精度 128；grid=16 原精度 64。

### 5.5 预乘 RGBA over 合成（_forward_strokes / _commit_strokes 共用）

```
out_rgb = stroke_rgb_premul + (1 - A) · canvas_rgb_premul
out_A   = A + (1 - A) · canvas_A
```

`_commit_strokes` 批化 forward（`_COMMIT_CHUNK` 笔一组）+ 组内逐笔串行 over 保序
（预乘 over 不可交换，重叠区按笔顺序叠加）。

---

## 6. 现状耦合点（参数无关化的待办清单）

> 这些是「规范化 + 参数无关化」总目标要解耦的点。**本次「数据精简为切片+置换」重构
> 已处理 #1 / #2 / #5**（见各行"已处理"标注）；#3 / #4 / #6 / #7 / #8 / #9 / #10 仍未解耦。

| # | 耦合点 | 现状 | 处理状态 / 参数无关化方向 |
|---|---|---|---|
| 1 | **CoarseToFine config 硬编码** | ~~`n_target/factor/cap` 写死在 `__init__`，无构造参数~~ | ✅ 已处理：提为构造参数（带默认值，向后兼容） |
| 2 | **11 维参数布局假设** | `params_to_full` 仍硬编码 11 维布局（保留为便利方法） | ✅ 已处理（数据结构层）：`Level` 参数维度无关，不感知通道语义；`params_to_full` 保留为"11 维布局已知时的便利方法"，内部读 `transform`，与 Level 解耦 |
| 3 | **4 通道 RGBA 写死** | 全链路 `shape[1]==4` 断言；over 合成硬编码 4 通道 | 通道数参数化或由笔刷侧决定 |
| 4 | **等分网格 affine** | `a=region_norm/W, b=c=0`，仅支持轴对齐等分切片 | 支持任意仿射切片（affine_grid 风格） |
| 5 | **device 不一致** | ~~`affine_full` 在 CPU，`image` 在 target device，靠 `params_to_full` 的 `.to` 对齐~~ | ✅ 已处理：`_slice` 里 affine 建 `device=img.device` → `transform` 全程在 target device |
| 6 | **降采样方法固定** | `F.adaptive_avg_pool2d` | 降采样器可换（area/nearest/...） |
| 7 | **main 内联优化环境** | reparam/init_raw/_forward_strokes/_commit_strokes/三层循环全在 main.py | 抽成独立模块，与空间类解耦 |
| 8 | **ErrorMap 字段名耦合** | `init_raw` 直接读 `em.top_centroid` 等具名字段 | 矩特征走稳定中间表示（dataclass/dict 契约） |
| 9 | **canvas_full 单一载体** | 实色 / 风格化共用同一 canvas_full（commit 行二选一） | 风格化为可插拔 commit 策略 |
| 10 | **绝对路径硬编码** | `image_input` 里 target/canvas/output 绝对路径 | 路径参数化 |

---

## 7. 边界与不变量（现状）

- **四维范式**：进入计算空间后全程 `(B,4,H,W)`，导入边界（image_input）三维→四维，导出边界（viewer）四维→HWC。
- **不做防御性编程**：target None / 维度错 → 自然崩，不 guard；任一全局量为 None 下游自然崩。
- **不读图入上下文**：main 的 `plt.show` 弹窗展示切片/画布，不把 PNG 塞进 agent 上下文（viewer 同理，headless 落盘但不读回）。
- **target/canvas 同切法**：`_slice` 对 target 与同尺寸 canvas 切出几何完全一致——这是断点重续 / 画回全图不错位的前提。
- **target = 监督量，canvas = 断点载体，皆 detach**：不参与梯度（梯度只走笔刷参数 raw）。
- **iter_levels 粗→细**：`pyramid[0]` 最粗（grid=1），`pyramid[-1]` 最细（grid=n_target 规约值）。

---

## 8. 下一步（指向总目标，本子目标之后）

据第 6 节耦合清单，把 coarse_to_fine 的 IO 流程规范化为参数无关接口：
- 抽出**空间几何层**（切片/降采样/仿射/params 翻译）与**笔刷参数布局**解耦——CoarseToFine
  只管几何，不假设参数通道语义；
- config 参数化（`n_target/factor/cap` ✅ 已完成；降采样器 / 通道数 待办）；
- main 内联的优化环境抽成独立模块，循环结构显式化；
- ErrorMap 矩特征走稳定中间表示。

（具体重构方案待总目标阶段设计，不在本现状文档范围。）
