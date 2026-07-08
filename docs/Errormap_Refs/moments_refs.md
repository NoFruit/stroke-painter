# 图像矩（image moments）refs — errormap 第三部分（矩计算）实现基础

配套：本文件由 opensearch 检索 + Wikipedia/skimage/RLEMaskLib/moments.py 交叉确认得出，
作为 `error_map.py` 矩计算步骤（CCL top 块 → 块的质心/走向/轴长/颜色）的标准算法依据，减少凭印象写。

---

## 0. 源清单（已读，已交叉验证）

| 源 | 提供什么 | 可信度 |
|---|---|---|
| Wikipedia - Image moment | raw/central moments、协方差矩阵、orientation、eigenvalues、eccentricity 全套权威公式 | 权威定义 |
| RLEMaskLib (readthedocs) | orientation 的 **atan2 鲁棒版**、eigenvalues、eccentricity（与 cv2.moments 对齐） | 交叉验证 Wikipedia |
| skimage 0.26 regionprops 文档 | 属性语义定义（axis_major_length / orientation / eccentricity / moments_central / inertia_tensor） | 标准属性集 |
| shackenberg/Image-Moments-in-Python moments.py | raw/central/normalized moments 的 numpy 参考实现 | 参考实现 |
| forum.image.sc | axis length 换算提示 "r = 2·√(...)" | 辅助 |

**注**：skimage `_regionprops.py` 源码因 GitHub 路径 404（新版疑似拆分）未直读，axis_length 的换算系数
改用「均匀椭圆二阶矩」数学推导自洽给出（见 §4），不依赖 skimage 源码记忆。

---

## 1. 标准公式（Wikipedia 权威）

### 1.1 Raw moments（原始矩）

对灰度图 I(x,y)（标量场）：

```
M_ij = Σ_x Σ_y  x^i · y^j · I(x,y)
```

低阶常用：
- **M_00 = Σ I(x,y)** — 面积（二值图）或强度和（灰度图）
- M_10, M_01 — 一阶矩
- M_20, M_02, M_11 — 二阶矩
- M_30, M_03, M_21, M_12 — 三阶矩（本项目不需要，Hu 不变量才用）

### 1.2 面积与质心

```
area    = M_00
centroid = ( x̄, ȳ ) = ( M_10 / M_00 ,  M_01 / M_00 )
```

### 1.3 Central moments（中心矩，平移不变）

```
μ_pq = Σ_x Σ_y  (x - x̄)^p · (y - ȳ)^q · I(x,y)
```

用 raw moments 展开（避免再扫一遍图，O(1) 由 raw 算出）：

```
μ_00 = M_00
μ_10 = 0,  μ_01 = 0
μ_11 = M_11 - x̄·M_01  (= M_11 - ȳ·M_10)
μ_20 = M_20 - x̄·M_10
μ_02 = M_02 - ȳ·M_01
```

> 关键：**只需要 6 个 raw moments（M_00, M_10, M_01, M_11, M_20, M_02）** 就能算出
> centroid + 全部二阶中心矩。本项目算到二阶足够（三阶只 Hu 不变量用，不需要）。

### 1.4 归一化二阶中心矩（协方差矩阵用，Wikipedia 记 μ'）

> 注意区分两种归一化：
> - **μ'_pq = μ_pq / μ_00**（除一次）→ 构协方差矩阵 cov[I]，Wikipedia 用此
> - **ν_pq = μ_pq / μ_00^((i+j)/2+1)**（除 m00 的 (i+j)/2+1 次方）→ scale-invariant，skimage moments_normalized 用此，Hu 不变量用此
>
> 本项目算 orientation/eigen/eccentricity 用 **μ'（除一次）** 即可，与 RLEMaskLib/cv2 对齐。

```
μ'_20 = μ_20 / μ_00  =  M_20/M_00 - x̄²
μ'_02 = μ_02 / μ_00  =  M_02/M_00 - ȳ²
μ'_11 = μ_11 / μ_00  =  M_11/M_00 - x̄·ȳ
```

### 1.5 协方差矩阵（图像强度的二阶矩）

```
cov[I] = [ μ'_20   μ'_11 ]
         [ μ'_11   μ'_02 ]
```

特征向量 = 强度分布的主/次轴方向；特征值 = 沿该轴的方差（∝ 轴长的平方）。

---

## 2. Orientation（走向 / 主轴角度）

Wikipedia 原式（arctan，分母为 0 时不稳）：
```
Θ = ½ · arctan( 2·μ'_11 / (μ'_20 - μ'_02) )     要求 μ'_20 ≠ μ'_02
```

**RLEMaskLib 鲁棒版（atan2，本项目采用）**：
```
Θ = ½ · atan2( 2·μ'_11 ,  μ'_20 - μ'_02 )
```
- atan2 自动处理分母 0 与象限，无需特判
- 分子分母同除 μ_00 不变，故等价于 `½·atan2(2·μ_11, μ_20-μ_02)`（用未归一 central moment 也行，结果相同）
- skimage 定义：range [-π/2, π/2]，从第 0 轴（row/y）到主轴，逆时针为正

> 笔画走向用此 Θ：P0→P2 方向沿主轴。

---

## 3. Eigenvalues（特征值，∝ 轴长²）

Wikipedia 与 RLEMaskLib 一致（RLEMaskLib 用 a=μ_20/m_00, b=μ_11/m_00, c=μ_02/m_00，即 μ'）：

```
λ₁ = ½·( (μ'_20 + μ'_02) + √( 4·μ'_11² + (μ'_20 - μ'_02)² ) )   ← 大特征值（主轴）
λ₂ = ½·( (μ'_20 + μ'_02) - √( 4·μ'_11² + (μ'_20 - μ'_02)² ) )   ← 小特征值（次轴）
```

---

## 4. Eccentricity 与 axis length

### 4.1 Eccentricity（Wikipedia 权威）
```
eccentricity = √( 1 - λ₂/λ₁ )      ∈ [0, 1)
  0 = 圆，接近 1 = 高度细长
```

### 4.2 axis length（均匀椭圆二阶矩推导，skimage 源码未直读故自洽推导）

**推导**：均匀密度椭圆，半轴 a(major)≥b(minor)，沿坐标轴对齐，密度 1：
- 面积 M_00 = π·a·b
- ∫∫_ellipse x² dx dy = π·a³·b/4，归一（除面积）→ μ'_major = a²/4
- 故 **a = 2·√(μ'_major) = 2·√λ₁**，同理 b = 2·√λ₂

换算：
| 量 | 公式 | 含义 |
|---|---|---|
| 半轴长 a (major) | `2·√λ₁` | 主轴半长（半径） |
| 半轴长 b (minor) | `2·√λ₂` | 次轴半长（半径） |
| 全轴长（直径） | `4·√λ₁` / `4·√λ₂` | skimage `axis_major_length` = `4·√λ_major`（全轴） |

> forum.image.sc 提示 "r = 2·√(...)" = 半轴；skimage `axis_major_length` = 全轴 = `4·√λ`。
> 本项目给笔画 init 建议：**主轴半长 a ≈ 笔画半长（P0→中心 或 P2→中心 距离）**，用 `2·√λ₁` 即可。

---

## 5. skimage regionprops 属性对照（语义对齐）

本项目矩计算输出应对齐这些标准属性名（方便对照/复用语义）：

| regionprops 属性 | 本项目对应 | 公式来源 |
|---|---|---|
| `centroid` | 质心 (x̄, ȳ) | §1.2 |
| `area` | M_00 | §1.1 |
| `moments_central` | μ_pq | §1.3 |
| `moments_normalized` | ν_pq（scale-invariant，本项目非必须） | §1.4 注 |
| `inertia_tensor` | cov[I] | §1.5 |
| `orientation` | Θ | §2 atan2 版 |
| `eccentricity` | √(1-λ₂/λ₁) | §4.1 |
| `axis_major_length` | 4·√λ₁（全轴） / 2·√λ₁（半轴） | §4.2 |
| `axis_minor_length` | 4·√λ₂（全轴） / 2·√λ₂（半轴） | §4.2 |

---

## 6. 本项目 errormap 应用子集（要算哪些）

errormap top 块（`_score_regions` 选出的 `top_label`）→ 对该块算矩特征，给笔画参数 init 建议：

| 笔画参数 | 矩特征 | 映射 |
|---|---|---|
| P0/P1/P2（位置+走向） | centroid + orientation Θ | 质心=笔画中心；Θ=P0→P2 方向 |
| 笔画长度 | 半轴 a = 2·√λ₁ | a ≈ 笔画半长 |
| r（粗度） | 半轴 b = 2·√λ₂ | b ≈ 笔画粗度 |
| c（颜色） | target RGB 在块内均值 | 见 §7 |
| alpha | （暂不从矩算，后续/固定） | — |

### 6.1 权重选择决策（待定，需用户拍板）

矩公式 `M_ij = Σ x^i y^j · I(x,y)` 中的 I(x,y) 用什么：

- **方案 A：binary mask 权重**（I = `labels==top_label`，0/1）
  - 算的是「块形状的几何矩」——描述块本身长什么样
  - 与 CCL 分块语义一致（块是几何连通区域）
  - → 适合描述形状/大小/流向（用户说的"形状、大小、流向"）
- **方案 B：intensity 权重**（I = `abs_diff` RGB mean）
  - 算的是「差异强度分布的矩」——描述差异集中在块里哪
  - 中心偏向差异最强的位置
  - → 适合描述"差异质心"而非"几何质心"

**建议**：形状维（centroid/orientation/轴长）用 **方案 A（binary mask）**，因为要的是"这块面积本身的形状"；
颜色维单独从 target 采样（见 §7）。此为开放决策，实现前与用户确认。

---

## 7. 颜色采样（块的颜色 c）

不在矩公式里，单独算：对 top 块区域，在 **target** 上取 RGB 均值：

```
c_block = mean over {(x,y) : labels==top_label} of target_rgb(x,y)     # (3,) RGB
```

- 用 target（不是 canvas）——init 建议要补成 target 的颜色
- `diff_direct`（target-canvas 有向差）的块均值可作为「该补多少色」的辅助，但 c 本身取 target 均值
- torch 批：scatter_add(target_rgb, label) / area，同 §8

---

## 8. torch 批实现要点（slice 间并行 + 块间 scatter 并行）

输入：`labels (B,1,ph,pw)` kornia 不连续大标签，`top_label (B,)`，`intensity (B,1,ph,pw)` 或 binary mask。

核心：**一次 scatter_add 算出每个 top 块的 6 个 raw moments**，再 O(1) 推 centroid/central/eigen/orientation。

```python
# 1. 取 top 块 mask（B,ph,pw）
top_mask = (labels[:,0] == top_label[:,None,None])          # bool, (B,ph,pw)

# 2. 权重场 W（方案 A: mask.float；方案 B: intensity）
W = top_mask.float()                                        # (B,ph,pw)

# 3. 坐标场（per-slice 共享，ph×pw）
ys, xs = torch.meshgrid(arange(ph), arange(pw), indexing='ij')   # 各 (ph,pw)

# 4. 6 个 raw moments —— 每块一个标量，scatter over 块
#    因为每 slice 只关心 1 个 top 块，直接按 B 维 sum（mask 已隔离该块）
M00 = (W).sum(dim=(1,2))                  # (B,)
M10 = (xs * W).sum(dim=(1,2))             # (B,)
M01 = (ys * W).sum(dim=(1,2))             # (B,)
M11 = (xs*ys * W).sum(dim=(1,2))          # (B,)
M20 = (xs*xs * W).sum(dim=(1,2))          # (B,)
M02 = (ys*ys * W).sum(dim=(1,2))          # (B,)

# 5. centroid + central moments（elementwise，已并行）
xc = M10 / M00.clamp(min=1)
yc = M01 / M00.clamp(min=1)
mu20 = M20 - xc*M10
mu02 = M02 - yc*M01
mu11 = M11 - xc*M01                        # = M11 - yc*M10

# 6. 归一化（μ' = μ/μ_00）+ eigen + orientation + axis
mu20p = mu20 / M00.clamp(min=1)
mu02p = mu02 / M00.clamp(min=1)
mu11p = mu11 / M00.clamp(min=1)
theta = 0.5 * torch.atan2(2*mu11p, mu20p - mu02p)        # (B,) 走向
lam1 = 0.5*((mu20p+mu02p) + torch.sqrt(4*mu11p**2 + (mu20p-mu02p)**2))   # 主轴
lam2 = 0.5*((mu20p+mu02p) - torch.sqrt(4*mu11p**2 + (mu20p-mu02p)**2))   # 次轴
a = 2*torch.sqrt(lam1.clamp(min=0))                       # 半轴长（笔画半长）
b = 2*torch.sqrt(lam2.clamp(min=0))                       # 半轴短（粗度）
ecc = torch.sqrt(1 - lam2/lam1.clamp(min=1e-12))          # 离心率
```

要点：
- **slice 间天然并行**：所有 (B,) 张量一次算，B 维并行（用户确认的并行模型）
- **块内**：每 slice 只 1 个 top 块，mask 隔离后直接 sum 即可，**不需要 bincount/unique**（比 `_score_regions` 更简单——`_score_regions` 要算所有块所以需 bincount，这里只 top 块，mask+sum 够）
- 坐标场 meshgrid 只需建一次（ph,pw 与 slice 无关），广播到 B
- 全程 detach（errormap 是信息类，不进 backward）

---

## 9. refs 链接

- Wikipedia Image moment: https://en.wikipedia.org/wiki/Image_moment
- RLEMaskLib compute-moments: https://rlemasklib.readthedocs.io/en/latest/howto/compute-moments.html
- skimage regionprops API: https://scikit-image.org/docs/stable/api/skimage.measure.html
- skimage regionprops 示例: https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_regionprops.html
- moments.py 参考实现: https://github.com/shackenberg/Image-Moments-in-Python/blob/master/moments.py
- Intel IPP Image Moments: https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2022-2/image-moments.html
- Hu 1962 原始论文: doi:10.1109/TIT.1962.1057692

---

## 10. 待确认（实现前问用户）

1. **权重方案 A vs B**（§6.1）：形状矩用 binary mask（推荐 A），还是 intensity 加权？
2. **轴长取半轴还是全轴**（§4.2）：笔画长度映射用 `2√λ₁`（半轴，推荐）还是 `4√λ₁`（全轴）？
3. **矩算对象**：只对 `top_label`（每 slice 1 块）算，还是对全部块算后选 top？（推荐只算 top 块，省算力，§8 实现即此）
4. **alpha**：是否也从矩/块特征给建议？（当前暂不算，§6）
