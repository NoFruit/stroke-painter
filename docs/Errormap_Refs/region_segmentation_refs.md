# error_map 区域归类算法选型方案 — 替换二值化+CCL

配套：本研究由 opensearch 检索 + 论文/库验证得出，解决"二值化阈值 τ 截断导致信息丢失 + 纯色块卡阈值给不出建议"问题。
`moments_refs.md` 是矩计算 refs；本文件是**分块归类**的 refs（替代 _binarize + _ccl）。

---

## 1. 问题重新表述

> 在单通道差异图上，把"值相近且位置相邻"的像素归为同一区域，输出每像素区域标签 (B,1,H,W long)。
> 区域数由数据决定，**不靠二值化阈值截断**。"该补多少"用区域均值（连续量）表达，而非 0/1 过阈值。

这就是 **SLIC 超像素**的定义语义：(位置 + 值) 联合特征空间的 k-means，位置加权保证紧凑。

---

## 2. 算法候选（已验证库支持）

| 算法 | 批GPU | 单通道 | torch 库 | 复杂度 | 结论 |
|---|---|---|---|---|---|
| **SLIC / k-means in (x,y,v)** | ✅ | ✅ | ✅ **torch_kmeans** (jokofa, PyPI) + kornia CCL | O(N·K·iter) | **推荐** |
| SSN / 可微 SLIC (diffSLIC) | ✅ | ⚠️ RGB 取向 | ⚠️ 研究仓库非 pip | O(N·K·iter) | 备选，改造量大 |
| Felzenszwalb 图分割 | ❌ | ✅ | ❌ 无 torch（仅 skimage CPU / CUDA C++） | O(E·α) | 排除：无 GPU 库 |
| Watershed 分水岭 | ❌ | ✅ | ❌ cucim 未实现(issue#89) | O(N) | 排除：无 torch 批库 |
| Mean shift / Quick shift | ❌ | ✅ | ❌ 仅 skimage CPU | O(N²) | 排除：无 GPU 库、计算重 |
| Region growing | ❌ 串行本质 | ✅ | ❌ 无 GPU 库 | O(N) | 排除：本质串行 |

**关键排除依据（已验证）**：
- **cucim**：Windows 原生不支持（issue #928），需 WSL2；watershed 在 issue #89 是未实现追踪项 → 出局
- **kornia**：contrib 只有 `connected_components`(CCL)，无 SLIC/超像素/watershed/felzenszwalb
- **pykeops**：Windows 装不了（本项目已验证）→ 排除一切依赖它的方案
- **NVlabs/ssn_superpixels**：Caffe+Cython，非纯 torch，非 pip → 排除

---

## 3. 推荐：SLIC 语义聚类（torch_kmeans + kornia CCL 兜底）

### 为什么 torch_kmeans

唯一同时满足全部硬约束的维护库（API 已验证 readthedocs）：
- 纯 torch 张量算子，**GPU + 原生 batch**：`forward(x)` 输入 `(BS,N,D)`，`fit_predict` 返回 `(BS,N)` LongTensor
- `k` 可传 **per-instance 张量 `(BS,)`** → 每个 slice 区域数不同（区域数不固定）
- 默认 `LpDistance(p_norm=2)`（欧氏），支持自定义 `BaseDistance` → SLIC 位置加权距离
- 可传初始中心 `(BS,K,D)` → SLIC 网格初始化
- 附 `SoftKMeans`（可微）/ `ConstrainedKMeans`（带权重）备用

### 碎片化兜底

k-means 标签可能给同值但空间分离的像素打同一号（非连通）。
**已有的 kornia CCL** 正好兜底：在 k-means 标签图上再跑一遍 CCL，把同号但不连通的块拆成独立区域。
两库协同，全程 torch+GPU+batch。

### 解决"纯色卡阈值"边缘 case

紫色块 RGB=(0.741,0,0.737)、差异均匀 0.493 < τ=0.5：
- 该块所有像素 v≈0.493、彼此相邻 → k-means 聚成**同一簇**，标签图上是一整块
- **不被丢弃**（无阈值截断这步）
- 区域均值 0.493 → 连续量"该补多少"的建议强度（旧流程只能给 0/1）
- 即使整图差异低，SLIC 仍产出若干区域并按均值排序，"最该补的"始终浮现

---

## 4. 新流程结构（替换 _combine_channels + _binarize + _ccl）

伪代码级（非可运行代码）：

```
输入: diff_map (B,1,H,W) float[0,1]   # |target-canvas| RGB 三通道均值差异图

# --- 步骤A: 构造每像素联合特征（替代 _combine_channels 角色）---
xs, ys = meshgrid(W, H)                          # 各 (H,W)
S = sqrt(H * W / K)                              # SLIC 期望超像素边长（位置加权尺度）
pos_x = xs / S                                   # (H,W)
pos_y = ys / S
feat = stack([pos_x, pos_y, diff_map], dim=-1)   # (B,H,W,3)  单通道值，无需合并多通道
feat = feat.reshape(B, H*W, 3)                   # (BS, N, D) 供 torch_kmeans

# --- 步骤B: 聚类（替代 _binarize + _ccl）---
kmeans = KMeans(n_clusters=K, distance=LpDistance(p_norm=2),
                init_method='k-means++', max_iter=10)
labels = kmeans.fit_predict(feat)                # (B, H*W) long
labels = labels.reshape(B, 1, H, W)             # 区域标签图

# --- 步骤C: 连通性兜底（复用已有 kornia CCL）---
labels = kornia.contrib.connected_components(labels, num_iterations=...)  # 拆同号不连通块

# --- 步骤D: 矩特征统计（已写好的 scatter_add，不变）---
# 按标签 scatter_add 算：区域质心/走向/轴长/均值差异/面积
# 选差异均值最大（或差异总量=均值×面积最大）的区域 → 笔画初始化建议
```

**三步替换映射**：
| 旧 | 新 |
|---|---|
| `_combine_channels` | 步骤A（单通道，无需合并；构造 (x,y,v) 特征） |
| `_binarize`（阈值τ） | **删除** —— 无截断，区域均值即连续"该补多少" |
| `_ccl`（二值连通） | 步骤B(torch_kmeans) + 步骤C(kornia CCL 兜底碎片) |

---

## 5. 待你决策的点（subagent 未落地）

1. **K 取值**：建议过分割 K≈20-50 再按均值排序选 top-1，或 per-slice 自适应
2. **是否 per-slice 不同 K**：torch_kmeans 支持 per-instance k 张量 (BS,)
3. **位置权重 S 的标定**：S = √(HW/K) 是 SLIC 经典，可能需调
4. **是否加相邻簇合并（agglomerative）**：逼近 Felzenszwalb 自适应区域数
5. **打分标准变不变**：从"块内 intensity 均值"（旧）→ 仍可用区域 intensity 均值；或改成 intensity 总量（均值×面积）

---

## 6. refs（subagent 已验证）

**主方案库**
- torch_kmeans (jokofa) — PyPI: https://pypi.org/project/torch-kmeans/  |  GitHub: https://github.com/jokofa/torch_kmeans  |  API: https://torch-kmeans.readthedocs.io/en/latest/api/torch_kmeans.html
- kornia CCL（已在用）: https://www.kornia.org/tutorials/nbs/connected_components.html

**SLIC / 超像素算法基础**
- SLIC 原始论文 (Achanta et al.): https://www.epfl.ch/labs/ivrl/research/slic-superpixels/
- SSN 论文 (Jampani et al., ECCV2018) — arXiv: https://arxiv.org/abs/1807.10174
- perrying/diffSLIC（纯 torch 可微 SLIC，备选）: https://github.com/perrying/diffSLIC
- perrying/ssn-pytorch: https://github.com/perrying/ssn-pytorch

**对比/排除项**
- Felzenszwalb 论文: https://cs.brown.edu/people/pfelzens/papers/seg-ijcv.pdf
- skimage felzenszwalb/watershed(CPU): https://scikit-image.org/docs/stable/api/skimage.segmentation.html
- cucim Windows 不支持 issue #928: https://github.com/rapidsai/cucim/issues/928  |  cucim watershed 未实现 issue #89: https://github.com/rapidsai/cucim/issues/89
