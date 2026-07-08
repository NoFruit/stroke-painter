"""error_map.py — ErrorMap 信息类：按笔刷参数维度算误差引导图（不干预优化）。

定位
----
**信息类**。不进 loss / 不进 backward / 不改优化器。只算"差异引导图"并缓存到 self，
供调用方（后续 init 建议 / 优化中建议）消费。当前版本仅算图 + show 对照，看画对没画对。

笔刷参数的两个正交维度（决定 errormap 算几张图）
----------------------------------------------
可微笔画参数分两类，各自需要不同的 errormap 引导：

1. **形状维（几何）** — P0/P1/P2（位置、长度、走向）+ r（粗度）
   - P0/P1/P2 的点信息 → 笔画长度、走向（沿哪条线画）
   - r → 粗度（r 大=铺底，r 小=铺线）
   - 引导信号：**结构/空间**——哪里有边缘（走向）、哪里差最多（去哪/粗细）
   → 算：绝对差图（去哪）、target 梯度图（走向，Sobel 边缘结构）

2. **颜色维（物理量）** — c（3 维 RGB）+ alpha（1 维）
   - 主相关 loss = L1（像素差）；边缘处颜色有梯度
   - 引导信号：**有向差**——正=缺什么色（该补），负=画多了（该擦）
   → 算：有向差图 target-canvas（颜色采样方向）

与 loss 的关系（开放）
---------------------
当前版本仅 init 建议方向（未接）。loss 的 L1/OT/Grad/Area 已在算差异并回传，
errormap 不重复进 backward。后续可加"优化中建议"，但本阶段只算图 + show。

slice-slice（与优化同分辨率）
----------------------------
每波计算只与金字塔一层作主体：``compute`` 收该层同分辨率两批切片
``canvas_batch`` + ``target_batch``，输出各 errormap ``(B,1,ph,pw)``。
__init__ 持有 target 全图作源；切片工作由 c2f 做，ErrorMap 不自己切。

不迎合外部接口
--------------
errormap 写清自己要什么：compute 收两批同分辨率切片（canvas + target）。
不迁就 main 的 _forward_strokes 签名（pred/brush_alpha/patch_size 是渲染管线概念，
errormap 不管）。后续要 adaptor 是后续的事。

阶段性单测（不接入 main.py）
---------------------------
类内 main() 模拟 main 流程：target→c2f 切片→空白 canvas 切片（不画笔）→compute→plt.show。
管线 = 三张引导图（abs_diff/grad/diff_direct）+ 图二 SLIC 分区→打分选 top 块→5 个矩特征，
五列对照（target/intensity/labels/top块/top块+矩叠加）看分区与矩对不对。
"""

import math

import torch
import torch.nn.functional as F


# 亮度系数（与 GradientLoss._to_lum 同约定，Rec.601）
_LUM = (0.299, 0.587, 0.114)


def _to_lum(img: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) → (B,1,H,W) 亮度（标量场，结构/走向用）。"""
    r, g, b = _LUM
    return (r * img[:, 0] + g * img[:, 1] + b * img[:, 2])[:, None]   # (B,1,H,W)


def _sobel_xy(img: torch.Tensor):
    """(B,1,H,W) → (gx, gy)，Sobel 梯度。Sobel 1/8 归一（与 GradientLoss 同口径）。

    用 grouped conv：输入 (B,1,H,W) × 1 group × (1,1,3,3) 核 → (B,1,H,W)。
    """
    dev = img.device
    dt = img.dtype
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dt, device=dev) / 8.0
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dt, device=dev) / 8.0
    kx = kx.view(1, 1, 3, 3)
    ky = ky.view(1, 1, 3, 3)
    gx = F.conv2d(img, kx, padding=1)
    gy = F.conv2d(img, ky, padding=1)
    return gx, gy


def region_stats(labels: torch.Tensor, intensity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """逐批分块统计：块面积 + 块内 intensity 均值（批处理并行，无 B 循环）。

    用 batched scatter_add_ 替代 unique + remap + bincount：
    - scatter_add_ 对索引无连续性要求，直接操作 SLIC 原始标签
    - 一次 kernel 调用完成所有 B×N 像素的累加
    - 消除 B 级循环、消除 unique（O(N log N)）、消除 remap 表分配

    Args:
        labels   : (B,1,ph,pw) long, SLIC 区域标签（1..K，0 闲置；scatter 对值无连续性要求）
        intensity: (B,1,ph,pw) float, 单通道强度图（RGB abs_diff 均值 ∈ [0,1]）

    Returns:
        area : (B, L) float — 每标签的像素数（L = 全局 max_label + 1）
               area[b, label_value] = slice b 中标签 label_value 的面积
        score: (B, L) float — 每标签的 intensity 均值（intensity 已 mean 归一为 [0,1]）
               score[b, label_value] = 该块 intensity_mean
    """
    B, _, H, W = labels.shape
    dev = labels.device
    lbls = labels[:, 0]                         # (B, H, W)
    ints = intensity[:, 0]                      # (B, H, W)

    max_lbl = int(lbls.max())                   # 全局最大标签
    N = H * W
    flat_lbls = lbls.reshape(B, N)              # (B, N)
    flat_ints = ints.reshape(B, N).float()      # (B, N)
    ones = torch.ones_like(flat_ints)

    area = torch.zeros(B, max_lbl + 1, device=dev, dtype=torch.float32)
    diff_sum = torch.zeros(B, max_lbl + 1, device=dev, dtype=torch.float32)

    # batched scatter_add_ — 一次调用，无循环
    area.scatter_add_(dim=1, index=flat_lbls, src=ones)
    diff_sum.scatter_add_(dim=1, index=flat_lbls, src=flat_ints)

    # 排除背景（标签 0）
    area[:, 0] = 0
    diff_sum[:, 0] = 0

    # score = 区域内 intensity 均值。intensity 来自 _combine_channels=mean(三通道)，
    # 每像素已是 [0,1] 归一标量，区域和 / 面积即 [0,1] 均值（无额外 /3——sum 时代才需 /3）。
    score = diff_sum / area.clamp(min=1.0)  # (B, L)
    return area, score


# ------------------------------------------------------------------ #
# top 块矩特征（笔刷无关中间表示，给参数 init 建议）— refs moments_refs §6/§8
# 5 个高关联量，各自独立纯函数。前 4 个共享 6 个 raw moments，第 5 个 color 独立采 target。
# 权重方案 A：binary mask（块几何形状，非差异强度）——描述块本身长什么样，对 stamp 类型鲁棒。
# slice 间 B 维并行：每 slice 只 1 个 top 块，mask 隔离后直接 sum，无需 bincount/unique。
# ------------------------------------------------------------------ #
def block_raw_moments(top_mask: torch.Tensor, ys: torch.Tensor, xs: torch.Tensor):
    """top 块 raw moments（binary mask 权重）：6 个 (B,) 张量。

    标准定义（refs §1.1）：M_ij = Σ xⁱ yʲ · I(x,y)，I = top_mask（块几何形状）。
    只算到二阶（centroid/orientation/轴长 用），不算三阶（Hu 不变量才用）。
    全背景片（mask 全 False）→ M00=0 → 后续 clamp 防 nan，特征归 0。
    """
    W = top_mask.to(torch.float32)                          # (B,ph,pw)
    M00 = W.sum(dim=(1, 2))
    M10 = (xs * W).sum(dim=(1, 2))
    M01 = (ys * W).sum(dim=(1, 2))
    M11 = (xs * ys * W).sum(dim=(1, 2))
    M20 = (xs * xs * W).sum(dim=(1, 2))
    M02 = (ys * ys * W).sum(dim=(1, 2))
    return M00, M10, M01, M11, M20, M02


def feat_centroid(moments) -> torch.Tensor:
    """raw moments → 质心 (x̄, ȳ) = (M10/M00, M01/M00)，(B,2)。refs §1.2。"""
    M00, M10, M01 = moments[0], moments[1], moments[2]
    z = M00.clamp(min=1.0)
    return torch.stack([M10 / z, M01 / z], dim=1)           # (B,2) (xc, yc)


def feat_orientation(moments) -> torch.Tensor:
    """raw moments → 主轴走向 Θ = ½·atan2(2·μ'₁₁, μ'₂₀-μ'₀₂)，(B,) rad。refs §2。

    atan2 鲁棒版（自动处理分母 0，无需特判）。μ' = μ/μ₀₀（归一化二阶中心矩）。
    Θ w.r.t. col 轴(x)，range [-π/2, π/2]——主轴无向（180°对称），头尾方向需别处补。
    """
    M00, M10, M01, M11, M20, M02 = moments
    z = M00.clamp(min=1.0)
    xc = M10 / z
    yc = M01 / z
    mu20p = (M20 - xc * M10) / z
    mu02p = (M02 - yc * M01) / z
    mu11p = (M11 - xc * M01) / z                              # = (M11 - yc*M10)/z
    return 0.5 * torch.atan2(2.0 * mu11p, mu20p - mu02p)    # (B,)


def _eigvals(moments):
    """raw moments → 协方差矩阵特征值 (λ₁, λ₂)，各 (B,)。refs §3。"""
    M00, M10, M01, M11, M20, M02 = moments
    z = M00.clamp(min=1.0)
    xc = M10 / z
    yc = M01 / z
    mu20p = (M20 - xc * M10) / z
    mu02p = (M02 - yc * M01) / z
    mu11p = (M11 - xc * M01) / z
    common = torch.sqrt((mu20p - mu02p) ** 2 + 4.0 * mu11p ** 2)
    lam1 = 0.5 * (mu20p + mu02p + common)                    # 大特征值（主轴）
    lam2 = 0.5 * (mu20p + mu02p - common)                    # 小特征值（次轴）
    return lam1, lam2


def feat_axis_major(moments) -> torch.Tensor:
    """raw moments → 主轴半长 a = 2·√λ₁，(B,)（笔画半长）。refs §4.2。

    半轴（半径），非 skimage axis_major_length 全轴(4√λ)。clamp λ≥0 防 float 负值。
    """
    lam1, _ = _eigvals(moments)
    return 2.0 * torch.sqrt(lam1.clamp(min=0.0))             # (B,)


def feat_axis_minor(moments) -> torch.Tensor:
    """raw moments → 次轴半长 b = 2·√λ₂，(B,)（笔画粗度 r）。refs §4.2。"""
    _, lam2 = _eigvals(moments)
    return 2.0 * torch.sqrt(lam2.clamp(min=0.0))             # (B,)


def feat_color(top_mask: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
    """top 块内 target RGB 均值 c，(B,3)。refs §7。

    在 target（非 canvas）上采样——init 建议要补成 target 的颜色。
    全背景片 → (0,0,0)。
    """
    m = top_mask.to(torch.float32)[:, None]                  # (B,1,ph,pw) 广播到 RGB 通道
    z = top_mask.to(torch.float32).sum(dim=(1, 2)).clamp(min=1.0)  # (B,)
    csum = (target_rgb * m).sum(dim=(2, 3))                  # (B,3)
    return csum / z[:, None]                               # (B,3)


class ErrorMap:
    """误差信息类：按笔刷参数维度算误差引导图，缓存到 self（不干预优化）。

    compute 收两批同分辨率切片（canvas + target），算引导图分别缓存：
        - abs_diff   : |target - canvas| 逐通道绝对差（B,4,ph,pw，各通道 [0,1]，不归约）
        - grad_map   : |∇target|        target 边缘强度（形状维-走向，沿哪画）
        - diff_direct: target - canvas   RGB 有向差（颜色维，正=缺色该补）
        - labels     : SLIC 分区 + 同值簇合并后的区域标签（B,1,ph,pw long，1..K 子集，0 闲置无背景）
        - top_label/top_area/top_score : 每批差异均值最大的块（"最该补"，intensity_mean 评分）
        - top_centroid/orientation/axis_major/axis_minor/color : top 块矩特征
          （笔刷无关中间表示 → 参数 init 建议；refs moments_refs §6，5 个高关联量）
    """

    def __init__(self, target: torch.Tensor):
        """缓存 target 全图（detach）作源 + 尺寸参考。device 跟 target 走（输入跟随）。

        target : (1,3,H,W) RGB float [0,1]。compute 主路径收已切批，
                不直接用此全图（仅作源/尺寸/全图可视化备选）。
        """
        if target is None:
            raise RuntimeError("ErrorMap 强需求 target，got None。")
        t = target.detach().float()
        if t.dim() != 4:
            raise RuntimeError(f"target 需为 (1,3,H,W) RGB 四维，got {tuple(t.shape)}。")
        self.target = t                            # (1,3,H,W) on target.device（输入跟随）
        self.H, self.W = t.shape[-2], t.shape[-1]
        # ---- 图二（区域分区）超参（硬编码手改）----
        # SLIC 语义 k-means：RGB abs_diff 三通道均值 → 单通道强度 v ∈[0,1]，与位置联合聚类，
        # 无阈值无背景（替代旧 二值化+CCL，避免 τ 截断丢信息，如 slice 12 均匀块卡阈值）。
        self.n_clusters = 30      # 期望簇数 K（实际 K=gh*gw≥此值，网格对齐）。
                                  # 128² patch → K≈30，每簇约 550px（~23px 见方），过分割后选 top。
        self.slic_iters = 10      # k-means EM 收敛迭代数（assign↔recompute，固定轮数不检测收敛）。
        self.slic_compact = None  # 值权重 w（None→√K 平衡位置/值范围）。
                                  # 调大→值优先，区域贴差异轮廓；调小→空间紧凑趋纯网格。
        # ---- 合并后处理（_merge_clusters，FAST 路线，refs slic_merge_refs）----
        # SLIC 固定 K 过分割 → 相邻同值簇合并：均匀区→1 区，复杂区保留细分，区域数自适应。
        self.merge_q = 10.0       # τ 分位数百分位（每 slice 独立 τ=max(percentile(邻边μ差,q),eps)）。
                                  # 调小→更严合并（少并，保留细分）；调大→更松（多并，趋大区）。
        self.merge_eps = 1e-2     # τ 下限/噪声地板。均匀片邻边μ差≈0→τ=eps→全并；防退化图 τ=0。
        # 各维度引导图缓存（compute 后填充，detach）
        self.abs_diff: torch.Tensor | None = None       # (B,3,ph,pw) 分通道绝对差
        self.grad_map: torch.Tensor | None = None       # 形状维-走向（target 梯度强度）
        self.diff_direct: torch.Tensor | None = None    # 颜色维（有向差）
        # 图二缓存（compute_regions 后填充）
        self.labels: torch.Tensor | None = None          # (B,1,ph,pw) SLIC 区域标签（1..K，0 闲置无背景）
        self.top_label: torch.Tensor | None = None       # (B,) 每批最该补的块标签
        self.top_area: torch.Tensor | None = None        # (B,) 该块面积（像素数）
        self.top_score: torch.Tensor | None = None       # (B,) 该块差异均值（[0,1]，面积归一）
        # top 块矩特征（笔刷无关中间表示，compute_regions 末尾算，给参数 init 建议）
        # refs moments_refs §6：5 个高关联量，binary mask 权重（方案 A）
        self.top_centroid: torch.Tensor | None = None    # (B,2) 质心 (xc, yc) → 笔画中心
        self.top_orientation: torch.Tensor | None = None # (B,) 走向 Θ (rad) → P0→P2 方向
        self.top_axis_major: torch.Tensor | None = None  # (B,) 主轴半长 2√λ₁ → 笔画半长
        self.top_axis_minor: torch.Tensor | None = None  # (B,) 次轴半长 2√λ₂ → 粗度 r
        self.top_color: torch.Tensor | None = None       # (B,3) 块内 target RGB 均值 → 颜色 c
        # 私有瞬态缓存（compute 填，compute_regions 消费，非最终输出）
        self._target_rgb: torch.Tensor | None = None     # (B,3,ph,pw) 当前切片 target RGB

    # ------------------------------------------------------------------ #
    # 接口：compute 三张引导图，分别缓存到 self
    # ------------------------------------------------------------------ #
    def compute(self, canvas_batch: torch.Tensor, target_batch: torch.Tensor) -> "ErrorMap":
        """按笔刷参数两维度算三张误差引导图，缓存到 self，返回 self（链式）。

        收两批同分辨率切片（不迎合外部渲染管线签名）。每波只与金字塔一层作主体。

        Args:
            canvas_batch : (B,3,ph,pw) RGB 当前画布切片。
            target_batch : (B,3,ph,pw) RGB 目标切片（与 canvas 同 B、同 ph,pw）。

        Returns:
            self（三张图存 self.abs_diff / self.grad_map / self.diff_direct）。
        """
        cb = canvas_batch.detach().float()
        tb = target_batch.detach().float()

        # ---- 图一（形状维-去哪）：绝对差图 abs_diff（分通道，不归约）----
        # 标准 absolute difference image（ref: MATLAB imabsdiff / Wolfram ImageDifference /
        # OpenCV absdiff）：逐像素 |target - canvas|，**输出与输入同形状 (B,3,ph,pw)**，各通道
        # **独立、不合并、不归一**（ref 明确"channels compared separately"，无归一化）。
        #   - 每通道范围 [0,1]（不是 sum 到 [0,3]）
        #   - 归约（L1 求和 / 取范数 / argmax）推迟到后续需要单通道的图，不在图一做--保持 ref 标准。
        # target / canvas 皆 RGB（无 alpha），同表示直接相减，无需预乘对齐。
        self.abs_diff = (tb - cb).abs().detach()            # (B,3,ph,pw) 分通道绝对差，各 [0,1]

        # ---- 形状维-走向：|∇target| target 边缘强度 ----
        # target 自身结构 → 笔触沿边缘走（painterly 经典）。与 canvas 无关（结构是 target 固有）。
        tb_lum = _to_lum(tb)                          # (B,1,ph,pw) 亮度降维求梯度（结构走亮度）
        gx, gy = _sobel_xy(tb_lum)                    # 各 (B,1,ph,pw)
        self.grad_map = torch.sqrt(gx * gx + gy * gy + 1e-12).detach()   # (B,1,ph,pw)

        # ---- 颜色维：target - canvas RGB 有向差 ----
        # 正=缺该色（该补），负=画多了（该擦）。RGB 通道均值（颜色采样方向，统一并行）。
        self.diff_direct = ((tb - cb).mean(dim=1, keepdim=True)).detach()  # (B,1,ph,pw)

        # ---- 图二（形状维-分块）：RGB 通道 → SLIC 分区 → 打分选最该补的块 ----
        # alpha 不计入（RGB 三通道进分区）。SLIC 语义 ref 路径（见各私有方法 docstring）。
        self._target_rgb = tb.detach()                      # (B,3,ph,pw) color 采样源（compute_regions 用）
        self.compute_regions()

        return self

    # ------------------------------------------------------------------ #
    # 图二：分块（SLIC 分区 → 同值簇合并 → 打分），批处理
    # ------------------------------------------------------------------ #
    def compute_regions(self) -> "ErrorMap":
        """RGB abs_diff → SLIC 分区 → 同值簇合并 → 打分选最该补的块，缓存到 self。

        SLIC 语义（替代旧 二值化+CCL，无阈值无背景）：
          1) 合通道：RGB 三通道均值 → 单通道强度场 v ∈[0,1]（_combine_channels）
          2) 分区：(x/S, y/S, w·v) 联合特征空间批处理 k-means → 每像素区域标签 1..K（_slic_cluster）
          3) 合并：相邻同值簇（|μᵢ−μⱼ|<τ，τ 每 slice 自适应分位数）合成大区，区域数自适应——
             均匀区→1 区，复杂区保留细分（_merge_clusters，refs slic_merge_refs）
          4) 打分：块内 intensity 均值最大 = "最该补的块"（面积归一，避免大块偏向）
          5) 矩特征：对 top 块算 5 个高关联量（centroid/orientation/轴长/color），立刻得一组
             笔刷无关参数建议（refs moments_refs §6）。

        批处理：slice 间 B 维 + 片内 N 维两级并行（_slic_cluster bmm/scatter、
        _merge_clusters scatter/where 各 slice 并行，唯串行=min-flooding K 轮标签传递）。
        RGB 全通道进分区（无 alpha）。
        """
        ad_rgb = self.abs_diff                            # (B,3,ph,pw) RGB
        intensity = self._combine_channels(ad_rgb)        # (B,1,ph,pw) 合成单通道强度 v
        self.labels = self._slic_cluster(intensity)       # (B,1,ph,pw) long ∈[1,K]，0 闲置
        self.labels = self._merge_clusters(self.labels, intensity)  # 相邻同值簇合并，区域数自适应
        self.top_label, self.top_area, self.top_score = self._score_regions(
            self.labels, intensity)                        # 打分：块内 intensity 均值

        # ---- 收尾：top 块矩特征（5 个高关联量，笔刷无关 → 参数 init 建议）----
        # refs moments_refs §8：每 slice 只 1 个 top 块，mask 隔离后直接 sum 得 6 个 raw moments，
        # O(1) 推 centroid/central/eigen/orientation/轴长。slice 间 B 维并行（无 bincount）。
        # SLIC 无背景：top_label 恒≥1、top_mask 非空；clamp 仍作防御（M00≥1 天然满足）。
        ph, pw = self.labels.shape[-2:]
        tl = self.top_label[:, None, None]                 # (B,1,1) 广播
        top_mask = (self.labels[:, 0] == tl) & (tl > 0)    # (B,ph,pw) bool；bg 片 → 全 False
        ys, xs = torch.meshgrid(
            torch.arange(ph, device=self.labels.device, dtype=torch.float32),
            torch.arange(pw, device=self.labels.device, dtype=torch.float32),
            indexing='ij')                                # 各 (ph,pw)，ys=row 向下，xs=col 向右
        moments = block_raw_moments(top_mask, ys, xs)       # 6 × (B,)
        self.top_centroid = feat_centroid(moments)          # (B,2) → 笔画中心
        self.top_orientation = feat_orientation(moments)    # (B,) → P0→P2 方向
        self.top_axis_major = feat_axis_major(moments)      # (B,) → 笔画半长
        self.top_axis_minor = feat_axis_minor(moments)      # (B,) → 粗度 r
        self.top_color = feat_color(top_mask, self._target_rgb)  # (B,3) → 颜色 c
        return self

    def _combine_channels(self, t: torch.Tensor) -> torch.Tensor:
        """多通道 tensor → 单通道强度场（SLIC 分区前的降维，固定 mean）。

        RGB 三通道均值归一化 → 输出 v ∈ [0,1]（既作 SLIC 聚类值维 w·v，又作打分 intensity）。
        纯数据计算：不假设通道数/形状，给什么合什么。
        """
        if t.dim() < 2 or t.shape[1] <= 1:
            return t                                        # 单通道或无通道维，原样
        return t.mean(dim=1, keepdim=True)

    def _slic_cluster(self, intensity: torch.Tensor) -> torch.Tensor:
        """单通道强度图 (B,1,H,W) → SLIC 语义区域标签 (B,1,H,W) long ∈ [1,K]。

        替代旧 _binarize + _ccl：在 (x/S, y/S, w·v) 联合特征空间做批处理 k-means。
        SLIC 超像素语义——值相近+位置相近→同区域，**无阈值、无背景**（避免 τ 截断丢信息，
        如 slice 12 均匀块 0.493<0.5 整片被滤）。位置加权 S 保证簇空间紧凑；值权重 w
        （slic_compact）控制"贴差异轮廓 vs 空间紧凑"的权衡。输出 +1 偏移使 0 闲置 → 保持
        下游 region_stats 的 area[:,0]=0 与 compute_regions 的 tl>0 守卫语义（零改动复用）。

        不接 CCL 兜底：kmeans 簇可能空间不连通，但矩公式对任意像素集恒有定义
        （centroid/orientation/特征值均有限；退化情形 atan2 落 0、λ clamp 防 float 微负），
        属"松映射"非错误；位置加权已把不连通压到罕见轻微（refs region_segmentation_refs §3/§4）。

        批处理两级并行（无 Python 级 slice/像素循环）：
          - slice 间 B 维：每 slice 独立 K 个中心，bmm/scatter 各 slice 并行
          - 片内 N 维：argmin/scatter 一次算完全部像素
        唯一串行 = slic_iters 次 EM 收敛迭代（k-means 本质，torch_kmeans 同样）。

        Args:
            intensity: (B,1,H,W) float [0,1]，_combine_channels 产出的单通道强度 v。
        Returns:
            labels (B,1,H,W) long ∈ [1,K]，0 闲置（无背景）。
        """
        B, _, H, W = intensity.shape
        dev = intensity.device
        dt = intensity.dtype
        K_set = self.n_clusters
        gh = int(math.ceil(math.sqrt(K_set)))
        gw = int(math.ceil(K_set / gh))
        K = gh * gw                                   # 网格实际簇数（≥设定，gh×gw 对齐）
        S = math.sqrt(H * W / K)                       # 网格间距（位置归一尺度，SLIC 经典）
        w = self.slic_compact if self.slic_compact is not None else math.sqrt(K)

        v = intensity[:, 0]                            # (B,H,W)
        ys = torch.arange(H, device=dev, dtype=dt)
        xs = torch.arange(W, device=dev, dtype=dt)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')  # 各 (H,W) row=y, col=x
        N = H * W
        pos_x = gx.reshape(-1) / S                      # (N,) 位置归一（1 单位=1 网格间距）
        pos_y = gy.reshape(-1) / S                      # (N,)
        v_flat = v.reshape(B, N)                        # (B,N)
        feat = torch.stack([
            pos_x[None].expand(B, N),
            pos_y[None].expand(B, N),
            v_flat * w
        ], dim=-1)                                     # (B,N,3)  位置∈[0,~√K]，w·v∈[0,~√K]

        # 网格初始化中心：gh×gw 个均匀分布像素的真实特征（位置+该像素 v），非 k-means++
        gx_idx = torch.linspace(0, W - 1, gw, device=dev).round().long()    # (gw,) col 索引
        gy_idx = torch.linspace(0, H - 1, gh, device=dev).round().long()    # (gh,) row 索引
        centers_idx = (gy_idx[:, None] * W + gx_idx[None, :]).reshape(-1)   # (K,) flat 像素索引
        centers = feat[:, centers_idx]                  # (B,K,3)

        feat_sq = (feat * feat).sum(-1, keepdim=True)   # (B,N,1) 预算 f²（迭代不变）
        for _ in range(self.slic_iters):
            # assign: ||f-c||² = f² + c² - 2 f·cᵀ（避免实体化显式 (N,K) 距离对，省显存）
            c_sq = (centers * centers).sum(-1)          # (B,K)
            d2 = feat_sq + c_sq[:, None, :] - 2 * torch.bmm(feat, centers.transpose(1, 2))  # (B,N,K)
            lbl = d2.argmin(-1)                         # (B,N)
            # recompute: scatter_add 按簇求新中心；空簇（cnt=0）保留旧中心
            sum_f = torch.zeros(B, K, 3, device=dev, dtype=dt)
            sum_f.scatter_add_(1, lbl[..., None].expand(-1, -1, 3), feat)   # (B,K,3)
            cnt = torch.zeros(B, K, device=dev, dtype=dt)
            cnt.scatter_add_(1, lbl, torch.ones(B, N, device=dev, dtype=dt))  # (B,K)
            empty = (cnt == 0)[..., None]              # (B,K,1)
            new_c = sum_f / cnt.clamp(min=1.0)[..., None]   # 空簇→0（被 where 拒绝）
            centers = torch.where(empty.expand_as(centers), centers, new_c)

        return lbl.reshape(B, 1, H, W).long() + 1      # (B,1,H,W) ∈[1,K]，0 闲置

    def _merge_clusters(self, labels: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
        """SLIC 标签 → 相邻同值簇合并后的标签 (B,1,H,W) long ∈[1,K]子集。

        FAST 路线（refs slic_merge_refs）：固定 K 强制切 K 块，均匀区被切成空间网格而非语义区
        （slice 12 整片紫 v=0.493 → 30 网格块）。本方法把"相邻且 v 均值差 < τ"的簇合并成大区，
        区域数由内容自适应：均匀区→1 区，复杂区保留细分。

        判定 |μᵢ−μⱼ|<τ（标量绝对差），τ 每 slice 自适应 `max(percentile(邻边μ差, q), ε)`。
        合并组本身空间连通（merge_edge 仅存于空间相邻簇间，沿合并边的路径必是空间路径）→
        不需 CCL 拆连通。单簇空间不连通是 SLIC 遗留问题，沿用既定"艺术奖赏"哲学不处理（不加 CCL）。

        批处理两级并行（无 B 级 Python 循环）：scatter/bmm/where 各 slice 并行；
        唯一串行 = min-flooding 的 K 轮标签传递（K≈30 微秒级，g 单调非增下界1，K轮≥直径收敛）。
        输出 min-root id（非连续但≥1，0 闲置）——region_stats 对非连续 id 鲁棒，无需重排连续。

        Args:
            labels   : (B,1,H,W) long ∈[1,K]，_slic_cluster 输出。
            intensity: (B,1,H,W) float，单通道 v（合并所依值，与分区同源）。
        Returns:
            (B,1,H,W) long ∈[1,K]子集，0 闲置。各 slice 区域数 M≤K 自适应。
        """
        B, _, H, W = labels.shape
        dev = labels.device
        lbl = labels[:, 0]                              # (B,H,W)
        v = intensity[:, 0]                             # (B,H,W)
        N = H * W
        vf = v.reshape(B, N)
        K = int(self.n_clusters)
        gh = int(math.ceil(math.sqrt(K)))
        gw = int(math.ceil(K / gh))
        K = gh * gw                                     # 实际簇数（与 _slic_cluster 对齐）
        q = self.merge_q
        eps = self.merge_eps

        # ---- 1) 每簇 v 均值 μ (B,K)：scatter_add over 像素 ----
        idx = (lbl.reshape(B, N).clamp(min=1) - 1)      # (B,N) ∈[0,K-1]，clamp 防 0 污染末列
        sumv = torch.zeros(B, K, device=dev, dtype=vf.dtype).scatter_add_(1, idx, vf)
        cnt = torch.zeros(B, K, device=dev, dtype=vf.dtype).scatter_add_(
            1, idx, torch.ones(B, N, device=dev, dtype=vf.dtype))
        mu = sumv / cnt.clamp(min=1.0)                  # (B,K)；空簇 μ=0 不进邻接图，无害

        # ---- 2) 相邻簇对 → 邻接矩阵 A (B,K,K bool)：4 邻域 roll ----
        # ★最隐蔽 bug：torch.roll 回绕把图对边相连 → 均匀片全片误并。必须屏蔽回绕行/列。
        A = torch.zeros(B, K, K, device=dev, dtype=torch.bool)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            s = torch.roll(lbl, shifts=(dy, dx), dims=(1, 2))
            valid = torch.ones(B, H, W, device=dev, dtype=torch.bool)
            if dy > 0:
                valid[:, :dy] = False                   # 屏蔽 roll 回绕的行
            elif dy < 0:
                valid[:, dy:] = False
            if dx > 0:
                valid[:, :, :dx] = False                # 屏蔽 roll 回绕的列
            elif dx < 0:
                valid[:, :, dx:] = False
            bnd = (s != lbl) & valid                    # (B,H,W) 不同簇的相邻像素对
            nz = torch.nonzero(bnd, as_tuple=False)     # (P,3) [b,y,x]，跨批混合无所谓
            if nz.numel() > 0:
                bb, yy, xx = nz[:, 0], nz[:, 1], nz[:, 2]
                ii = (lbl[bb, yy, xx] - 1).clamp(0, K - 1)
                jj = (s[bb, yy, xx] - 1).clamp(0, K - 1)
                A.index_put_((bb, ii, jj), torch.ones(bb.numel(), device=dev, dtype=torch.bool))
        A = A | A.transpose(1, 2)                       # 对称化（每对两方向都置）
        diag = torch.arange(K, device=dev)
        A[:, diag, diag] = False                        # 去自环（3D 不能用 fill_diagonal_）
        mu_diff = (mu[:, :, None] - mu[:, None, :]).abs()   # (B,K,K) 对称 μ 差

        # ---- 3) τ 自适应分位数（每 slice 独立，变长→定形 sentinel 法）----
        # 不能用 torch.quantile（不支持 mask，会把非邻接对的 0/inf 算进去）。
        ti, tj = torch.triu_indices(K, K, offset=1, device=dev)   # (P,) P=K(K-1)/2 定常
        diffs = mu_diff[:, ti, tj]                     # (B,P) 全上三角 μ 差
        has_pair = A[:, ti, tj]                        # (B,P) 是否真相邻
        masked = diffs.where(has_pair, torch.full_like(diffs, float('inf')))  # 非邻接填 +inf
        ms, _ = masked.sort(dim=1)                      # (B,P) 升序，inf 堆末尾
        n_pair = has_pair.sum(dim=1)                    # (B,) 每 slice 相邻对数
        rank = (n_pair.float() * (q / 100.0)).long().clamp(0, ms.shape[1] - 1)  # (B,)
        tau = ms.gather(1, rank[:, None])[:, 0]         # (B,) 分位值
        tau = torch.where(n_pair == 0, torch.full_like(tau, eps), tau).clamp(min=eps)
        # 无相邻对（单簇独占整片）→ τ=eps；其余 clamp 防 0。

        # ---- 4) 门控 + min-flooding 算簇级 group_id（K 轮标签传递）----
        gate = mu_diff < tau[:, None, None]            # (B,K,K) |μ_i-μ_j|<τ，对称
        merge_edge = A & gate                          # (B,K,K) 相邻且可并，对称
        g = torch.arange(1, K + 1, device=dev, dtype=torch.long)[None].expand(B, K).clone()
        SENT = K + 1                                   # 大于任何合法 id 的哨兵
        for _ in range(K):                             # K 轮 ≥ 直径(K-1)，保证收敛
            nbr = torch.where(merge_edge, g[:, None, :].expand(B, K, K),
                              torch.full((B, K, K), SENT, device=dev, dtype=torch.long))
            g = torch.minimum(g, nbr.min(dim=2).values)   # 邻域最小根；单调降→收敛
        # g[b,i] = i 所在 merge-CC 的最小根 id（∈[1,K] 子集，非连续）

        # ---- 5) group_img（每像素组 id）→ 返回 ----
        bi = torch.arange(B, device=dev)[:, None, None].expand(B, H, W)
        group_img = g[bi, (lbl.reshape(B, H, W) - 1).clamp(min=0)]   # (B,H,W) long
        return group_img[:, None].long()               # (B,1,H,W) ∈[1,K]子集，0 闲置

    def _score_regions(self, labels: torch.Tensor, intensity: torch.Tensor):
        """SLIC 标签图 + 强度图 → 每批 top 块（块内差异均值最大的块），批处理并行。

        内部调用 region_stats（纯函数）完成全块统计，再选 top。
        SLIC 无背景：labels∈[1,K]、每片必有簇 → top_label 恒≥1, top_area>0。
        has_regions 守卫现恒真但保留（无害）。

        Args:
            labels   : (B,1,ph,pw) long, SLIC 区域标签 ∈[1,K]，0 闲置
            intensity: (B,1,ph,pw) 强度图（三通道 abs_diff 均值，与分区同源）
        Returns:
            top_label (B,) long / top_area (B,) float / top_score (B,) float
        """
        area, score = region_stats(labels, intensity)       # (B, L) × 2
        B = labels.shape[0]
        dev = labels.device

        has_regions = area.sum(dim=1) > 0                   # (B,) 有块则 true
        best = score.argmax(dim=1)                          # (B,) index = 原始标签
        best = torch.where(has_regions, best, torch.zeros_like(best))

        idx = torch.arange(B, device=dev)
        return best, area[idx, best], score[idx, best]

    # ------------------------------------------------------------------ #
    # 自测（不接入 main.py）：模拟 main 真实流程，show 三图对照
    # ------------------------------------------------------------------ #
    @classmethod
    def main(cls):
        """自测：target→c2f 切片→空白 canvas 切片→compute→plt.show 五列对照。"""
        import matplotlib.pyplot as plt
        from coarse_to_fine import CoarseToFine
        import image_input
        from image_input import img_input

        img_input.load()
        target = img_input.target
        print("=" * 64)
        print("ErrorMap 自测 — 倒二层，空白 canvas 切片对比 target")
        print("=" * 64)
        print(f"[target] {tuple(target.shape)} on {target.device}")

        c2f = CoarseToFine(target)
        li = len(c2f.pyramid) - 2
        lvl = c2f.pyramid[li]
        target_batch = lvl.image
        b = lvl.n_tiles
        print(f"[level {li+1}/{len(c2f.pyramid)}] grid={lvl.grid_n}x{lvl.grid_n}  "
              f"B={b}  patch={lvl.patch_hw}")

        canvas_full = torch.zeros_like(target)
        canvas_batch = c2f.slice(canvas_full)[li].image
        print(f"[canvas] 空白切片，不画笔")

        em = cls(target)
        em.compute(canvas_batch, target_batch)
        # 图一 abs_diff（分通道）
        print(f"[abs_diff] shape={tuple(em.abs_diff.shape)}  "
              f"R[{float(em.abs_diff[:,0].min()):.3f},{float(em.abs_diff[:,0].max()):.3f}]  "
              f"G[{float(em.abs_diff[:,1].min()):.3f},{float(em.abs_diff[:,1].max()):.3f}]  "
              f"B[{float(em.abs_diff[:,2].min()):.3f},{float(em.abs_diff[:,2].max()):.3f}]")
        # 图二 SLIC 分区 + 合并：(x/S,y/S,w·v) k-means → 相邻同值簇合并 → labels → 打分选 top 块
        print(f"[slic] k-means: K={em.n_clusters} iters={em.slic_iters} "
              f"compact={em.slic_compact}")
        print(f"[merge] FAST adjacent-same-value: q={em.merge_q} eps={em.merge_eps}")
        print(f"  labels: {tuple(em.labels.shape)} {em.labels.dtype}  "
              f"[{int(em.labels.min())},{int(em.labels.max())}]")
        # top 块（差异均值最大 = 最该补）+ 5 个矩特征统计；M=合并后该 slice 区域数（对比 K=30）
        for i in (0, 12, 24, 36, 48, 60):
            cx, cy = em.top_centroid[i].tolist()
            m_i = int(em.labels[i].unique().numel())   # 合并后区域数（应 <K，均匀区≈1）
            print(f"  slice[{i:2d}] M={m_i:2d} lbl={int(em.top_label[i])} "
                  f"area={int(em.top_area[i])} score={float(em.top_score[i]):.3f}  "
                  f"centroid=({cx:.0f},{cy:.0f}) θ={float(em.top_orientation[i]):.2f}  "
                  f"a={float(em.top_axis_major[i]):.1f} b={float(em.top_axis_minor[i]):.1f}  "
                  f"rgb=({float(em.top_color[i,0]):.2f},"
                  f"{float(em.top_color[i,1]):.2f},{float(em.top_color[i,2]):.2f})")

        # show：target / intensity(v) / labels / top块 / top块+矩叠加（质心+主次轴+颜色）
        picks = [0, 12, 24, 36, 48, 60]
        fig, axes = plt.subplots(len(picks), 5, figsize=(16, 3 * len(picks)), dpi=80)
        intensity = em._combine_channels(em.abs_diff)   # (B,1,ph,pw) v（分区+打分同源）
        for r, i in enumerate(picks):
            axes[r][0].imshow(target_batch[i].permute(1, 2, 0).cpu().numpy())
            axes[r][1].imshow(intensity[i, 0].cpu().numpy(), cmap="viridis", vmin=0, vmax=1)
            axes[r][2].imshow(em.labels[i, 0].cpu().numpy(), cmap="nipy_spectral")
            tl = int(em.top_label[i])
            top_only = (em.labels[i, 0] == tl).cpu().numpy().astype("uint8")
            axes[r][3].imshow(top_only, cmap="gray", vmin=0, vmax=1)
            # 第 5 列：top 块灰底 + 质心点 + 主轴(红)/次轴(蓝) + 颜色块(右上)
            axes[r][4].imshow(top_only, cmap="gray", vmin=0, vmax=1)
            if int(em.top_area[i]) > 0:
                cx, cy = em.top_centroid[i].tolist()
                th = float(em.top_orientation[i])
                a = float(em.top_axis_major[i])
                b = float(em.top_axis_minor[i])
                dx, dy = math.cos(th), math.sin(th)         # 主轴方向 (x=col, y=row)
                axes[r][4].plot([cx - a * dx, cx + a * dx],
                                [cy - a * dy, cy + a * dy], 'r-', lw=1.5)   # 主轴 长度2a
                axes[r][4].plot([cx - b * (-dy), cx + b * (-dy)],
                                [cy - b * dx, cy + b * dx], 'b-', lw=1.5)   # 次轴 长度2b
                axes[r][4].plot(cx, cy, 'g+', ms=10, mew=2)                  # 质心
                rgb = em.top_color[i].clamp(0, 1).cpu().numpy()
                axes[r][4].add_patch(plt.Rectangle((0, 0), 8, 8, facecolor=rgb))
            axes[r][0].set_title(f"#{i} target", fontsize=9)
            axes[r][1].set_title(f"#{i} intensity(v)", fontsize=9)
            axes[r][2].set_title(f"#{i} labels (slic)", fontsize=9)
            axes[r][3].set_title(f"#{i} top (area={int(em.top_area[i])} "
                                 f"score={float(em.top_score[i]):.2f})",
                                 fontsize=9)
            axes[r][4].set_title(f"#{i} moment:th={float(em.top_orientation[i]):.2f} "
                                 f"a={float(em.top_axis_major[i]):.1f} "
                                 f"b={float(em.top_axis_minor[i]):.1f}", fontsize=9)
            for ax in axes[r]:
                ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        plt.show()
        print("=" * 64)
        print("done")
        print("=" * 64)


if __name__ == "__main__":
    ErrorMap.main()
