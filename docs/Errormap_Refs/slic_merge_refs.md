# SLIC 过分割后相邻同值簇合并 — 方向性 ref

## 问题回顾
固定 K=30 的 SLIC 在均匀残差区（v 处处相同）退化为纯位置网格切分，需把
相邻且同值的小簇合并成 1 个大区，复杂区保留细分。区域数自适应。

## 1. 合并判定标准

三种主流，利弊：

| 判定 | 做法 | 利 | 弊 |
|---|---|---|---|
| **簇均值差** `|μ_i − μ_j| < τ` | RAG 边权 = 两区均值 Euclidean 距离；greedy 合最小边直到 ≥ τ | 最简单，单通道标量 v 直接用绝对差；K=30 时 RAG 极小，瞬间完成 | 忽略区内方差，可能把"均值相近但内部杂"的区误合 |
| **严格相等** `μ_i == μ_j` | τ≈0 的均值差特例 | 对"整片紫 v=0.493"直接全合，零误合 | 对量化噪声/抗锯齿敏感，近均匀区不合 |
| **边界强度 (RAG boundary)** | skimage `rag_boundary` + Sobel 边缘幅值作边权，`merge_hierarchical` 合最弱边界 | 利用图像梯度证据，天然区分真假边界 | **残差图本身就是差异信号**，Sobel 在其上语义混乱；过度复杂 |

**推荐：簇均值差 `|μ_i − μ_j| < τ`**。本场景 v 是单通道标量残差，
"均值差"就是最直接的语义度量；边界强度法是为自然图像设计的，在残差图上
反而不对路。区内方差可作为二级 guard（可选）：`|μ_i−μ_j|<τ 且
max(σ_i,σ_j)<σ_floor` 才合，防止"均值巧合相近但内部都杂"的两区误合。

参考：
- skimage `graph.merge_hierarchical(labels, rag, thresh, ...)` 贪心合最小权边直到
  所有权 ≥ thresh。mean-RAG 用 `rag_mean_color`；boundary-RAG 用
  `rag_boundary` + Sobel。  
  https://scikit-image.org/docs/stable/api/skimage.graph.html
  https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_boundary_merge.html
- meysam-safarzadeh/SuperPixelSegmentation_using_SLIC：RAG 边权 = 两超像素
  均值 Euclidean 距离，再做 hierarchical merge / N-cut。  
  https://github.com/meysam-safarzadeh/SuperPixelSegmentation_using_SLIC
- Improving SLIC by color-difference-based region merging（SLIC + 颜色差合并两阶段）。  
  https://link.springer.com/content/pdf/10.1007/s11042-023-17304-7.pdf

## 2. τ 标定

- **固定值**（skimage demo 用 0.08，手调）：不跨图稳健。
- **自适应 — 按簇间均值差分布的低分位数**：收集 RAG 所有权重
  `w_e = |μ_i − μ_j|`，取 `τ = percentile(w_e, q)`，q≈10。
  - 均匀图：w_e 几乎全 ≈0 → τ≈0 → 全合为 1 块。
  - 复杂图：w_e 分布拉开，低分位数仍小 → 只合真正相近的邻区，复杂区保留。
  - 这正好对上"均匀区合、非均匀区不合"的目标。
- **噪声地板兜底**：`τ = max(percentile(w_e, 10), ε)`，ε 取 v 量化步长
  （如 1e-3）或 `0.05·std(v)`，防退化图。

**推荐：τ = max(percentile(邻边权重, 10), 1e-3)**。纯均匀区退化到 ε 合成 1 块；
非均匀区分位数自适应抬升，不误合。比固定 τ 稳，比严格相等抗噪。

参考：
- Adaptive strategy for superpixel-based region-growing segmentation
  (arXiv 1803.06541)：提出鲁棒相似度 + 自适应超像素合并策略，核心思想就是
  阈值随区域间差分布自适应，而非固定。  
  https://arxiv.org/abs/1803.06541
- Global superpixel-merging via set maximum coverage（ESWA 2023）：综述现有
  region-merging 多依赖最近邻对 + 固定/局部阈值，指出全局覆盖更优但开销大；
  本场景 K=30 不需要那么重。  
  https://www.sciencedirect.com/science/article/pii/S0952197623013969

## 3. torch 批处理可行性

**结论：技术上可行，但 K=30 下不值得做 GPU，CPU 回退代价几乎为零。**

- **无现成 drop-in**：没有等价于 `skimage.graph.merge_hierarchical` 的官方
  torch 批处理实现。pytorch_geometric 能在 GPU 上跑图，但层次合并的迭代
  贪心循环在 GPU 上别扭，工程量大于收益。
- **可行思路（若真要 GPU）**：本场景不需要完整层次合并，只需"相邻且
  `|μ_i−μ_j|<τ` 的簇连通合并"——等价于在门控邻接上做连通分量：
  1. `scatter_mean` 算每区 μ（批处理，GPU 友好）。
  2. 4/8 邻域像素对算 `|v_p − v_q| < τ` 得门控邻接。
  3. 连通分量：`kornia.contrib.connected_components` 或几轮 label-propagation。
  这条路批处理友好，但 K=30 时 RAG 仅 ~30 节点 ≤~60 边，毫无加速意义。
- **CPU 回退代价**：`merge_hierarchical` 在 K=30 的 RAG 上是微秒级；
  即便批处理多张图，构建 RAG + 合并的开销远小于上游 SLIC k-means 本身。
  RAG 构建可用 `skimage.graph.rag_mean_color` 或自己按邻域对索引。

**推荐：直接用 skimage `rag_mean_color` + `merge_hierarchical`（CPU），τ 取上述
自适应分位数。** 只有当未来 K 上到数百、且 batch 极大时才考虑 GPU
连通分量路线。

参考：
- skimage RAG / merge_hierarchical（CPU，K=30 足够）。  
  https://scikit-image.org/docs/stable/api/skimage.graph.html
- kornia connected_components（GPU 连通分量回退选项）。  
  https://github.com/kornia/kornia
- DeepMerge (arXiv 2305.19787)：DL+RAG 区域合并，证明 RAG 是可学习的，
    但对本场景过重。  
  https://arxiv.org/abs/2305.19787

## 一句话推荐
**判定：相邻簇均值差 `|μ_i−μ_j| < τ`（单通道 v 绝对差，可选加区内 σ guard）；
τ：`max(percentile(邻边权重, 10), 1e-3)` 自适应；
实现：CPU skimage `rag_mean_color` + `merge_hierarchical`，K=30 下 GPU 不划算。**
