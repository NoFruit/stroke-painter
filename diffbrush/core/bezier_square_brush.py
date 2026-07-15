"""BezierSquareBrush — 单色、均匀粗细的二次 Bézier 方头笔刷（disk-SDF + 端点切割）。

与 :class:`BezierUniformBrush` 共享 disk-SDF 圆刷覆盖，然后在两端点施加半平面
sigmoid 切割，将半圆头切为方头。原理：::

    M_round(x) = 1 − ∏_k(1 − w_k·disk_k)         # 圆刷覆盖（同 BezierUniformBrush）
    M_square(x) = M_round(x) · cut_start(x) · cut_end(x)   # 端点切割

其中 cut_start/cut_end 是过 P0/P2、法向沿切线的半平面 sigmoid 掩码。

参数（归一化输入，flat 向量，共 11 维）::

    Θ = { P0, P1, P2, c, r, α }

    [0:2]   P0  起点坐标      (2,)   归一化 [0,1] → 像素 [0,W]×[0,H]
    [2:4]   P1  控制点坐标    (2,)   同上
    [4:6]   P2  终点坐标      (2,)   同上
    [6:9]   c   单色 RGB      (3,)   [0,1]
    [9:10]  r   均匀半径      (1,)   归一化 → 像素 r·min(H,W)
    [10:11] α   透明度        (1,)   [0,1]

渲染流程（可微）：
  1. 控制点 / 半径归一化 → 像素坐标。
  2. 弧长 L = ∫||B'(t)||dt（闭式解）；间隔 d = r·ρ。
  3. Sigmoid 软计数：w_k = σ(α_cnt·(L/d − k))，k=1..K。
  4. stamp k 置于弧长 s_k = k·d 处（弧长→t 反演 → B(t_k)）。
  5. soft-disk 并集覆盖 M_round(x) = 1 − ∏_k(1 − w_k·disk_k)。
  6. 端点半平面切割：M = M_round · cut_endcaps(...)。
  7. 输出 (color, alpha) 双通道：color = c 广播 (B,3,H,W)（非预乘纯色），
     alpha = α·M (B,1,H,W)（覆盖度）。
"""
from typing import Tuple

import torch
from torch import Tensor

from .brush_base import BrushBase, ParamLayout
from ..geom import AABBGeom, BezierGeom, CapsGeom, StampGeom
from ..utils.stamp import pixel_grid, cut_endcaps, render_coverage

PARAM_DIM = 11


class BezierSquareBrush(BrushBase):
    @classmethod
    def param_layout(cls) -> ParamLayout:
        """Bezier 笔刷 (11-dim) 语义通道布局：P0[0:2], P1[2:4], P2[4:6], c[6:9], r[9], α[10]。"""
        return ParamLayout(
            param_dim=PARAM_DIM,
            point_slices=((0, 2), (2, 4), (4, 6)),
            color_slice=(6, 9),
            radius_idx=9,
            alpha_idx=10,
        )

    @staticmethod
    def unpack_params(params: Tensor):
        """(B, 11) → P0, P1, P2, c, r, alpha，各形状 (B, *)。"""
        P0 = params[:, 0:2]
        P1 = params[:, 2:4]
        P2 = params[:, 4:6]
        c = params[:, 6:9]
        r = params[:, 9:10]
        alpha = params[:, 10:11]
        return P0, P1, P2, c, r, alpha

    def __init__(self):
        # 固定数值超参（design_paradigm：作为 self 变量硬编码，不外置、不进 config）。
        self.sigmoid_alpha = 100.0       # 软计数 sigmoid 陡度
        self.max_stamp_K = 100          # 候选 stamp 数上界
        self.disk_softness = 100.0       # soft-disk 边缘陡度（无量纲，相对半径）
        self.stamp_spacing_rho = 0.5    # stamp 间隔比例 ρ：d = r·ρ（ρ<1 → 重叠覆盖）
        self.arc_length_grid = 256      # 弧长反演（arc_length_to_t 的 M）网格分辨率
        self.endcap_sharpness = 100.0   # 端点切割 sigmoid 陡度 β
        self.aabb_pad = 0.1            # AABB 正方形化外扩比例（吸收软过渡 / 数值误差）

    def forward(self, params: Tensor, patch_size) -> Tuple[Tensor, Tensor]:
        """归一化参数 -> (color, alpha)。

        新风格（geom 包）：曲线几何走 BezierGeom，stamp 参数走 StampGeom，
        端点切线走 CapsGeom；绘制侧（render_coverage / cut_endcaps）不变。

        Args:
            params: ``(B, 11)`` 或 ``(11,)`` 归一化参数（见模块 docstring）。
            patch_size: ``(H, W)`` patch 尺寸。

        Returns:
            (color, alpha)：color = c 广播 (B,3,H,W)（非预乘纯色），
            alpha = α·M (B,1,H,W)（覆盖度）；梯度均可通到 params。
        """
        if params.dim() == 1:
            params = params.unsqueeze(0)
        device = params.device
        dtype = params.dtype
        H, W = patch_size

        P0, P1, P2, c, r, alpha = self.unpack_params(params)

        scale = torch.tensor([float(W), float(H)], device=device, dtype=dtype)
        ref = float(min(H, W))
        P0p = P0 * scale
        P1p = P1 * scale
        P2p = P2 * scale
        r_px = r * ref                  # (B, 1) 方框半轴长（像素）

        # ---- 几何（r 无关）：BezierGeom 缓存系数 / 弧长，弧长采样 ----
        bg = BezierGeom(P0p, P1p, P2p)
        # ---- stamp 参数（几何无关）：L + r_px -> s_targets / weights ----
        s_targets, weights = StampGeom.setup(
            bg.length, r_px, self.stamp_spacing_rho, self.max_stamp_K, self.sigmoid_alpha
        )
        centers = bg.sample(s_targets, self.arc_length_grid)   # (B, K, 2)
        sg = StampGeom(centers, weights, r_px)                 # 绘制前夕的几何无关数据

        # ---- 绘制侧（不动）：soft-disk 并集覆盖 ----
        grid_coords = pixel_grid(H, W, device, dtype)           # (H, W, 2)
        coverage_round = render_coverage(
            sg.centers, sg.r_px, sg.weights, grid_coords, self.disk_softness
        )  # (B, H, W)

        # ---- 端点切线 -> CapsGeom（r 无关，仅 square）----
        t0 = torch.zeros(params.size(0), 1, device=device, dtype=dtype)
        t1 = torch.ones(params.size(0), 1, device=device, dtype=dtype)
        d0 = bg.derivative(t0)[:, 0, :]                          # (B, 2) B'(0)
        d1 = bg.derivative(t1)[:, 0, :]                          # (B, 2) B'(1)
        u0 = d0 / (torch.linalg.norm(d0, dim=-1, keepdim=True) + 1e-12)
        u2 = d1 / (torch.linalg.norm(d1, dim=-1, keepdim=True) + 1e-12)
        caps = CapsGeom(u0, u2)

        coverage = cut_endcaps(
            coverage_round, grid_coords, P0p, P2p, caps.u0, caps.u2, self.endcap_sharpness
        )  # (B, H, W)

        A = alpha.unsqueeze(-1) * coverage                # (B, H, W) 覆盖度 = α·M
        color = c[:, :, None, None].expand(-1, -1, H, W)  # (B, 3, H, W) 纯色（非预乘）
        alpha_out = A[:, None, :, :]                      # (B, 1, H, W) 覆盖度
        return color, alpha_out

    def forward_fast(self, params: Tensor, patch_size):
        """快速版 forward：段1 geom 几何 + aabb_info，段2 AABB 局部渲染 + 贴回。

        段1 用 geom 包产出几何：BezierGeom（系数/弧长/曲线 AABB/弧长采样）、
        StampGeom（采样集 + 半径）、AABBGeom（曲线 AABB + r -> tube/square/padded/
        integer 整数框）、CapsGeom（端点切线 u0/u2）。段2 由
        :meth:`_forward_aabb_patch_fast` 在 AABB 局部空间渲染圆刷覆盖，反 scatter
        贴回原 patch，最后叠加颜色与透明度。绘制侧（_forward_aabb_patch_fast）不变。

        Args:
            params: ``(B, 11)`` 或 ``(11,)`` 归一化参数（见模块 docstring）。
            patch_size: ``(H, W)`` patch 尺寸。

        Returns:
            (color, alpha)：color = c 广播 (B,3,H,W)（非预乘纯色），
            alpha = α·M (B,1,H,W)（覆盖度）。
        """
        if params.dim() == 1:
            params = params.unsqueeze(0)
        device = params.device
        dtype = params.dtype
        H, W = patch_size

        P0, P1, P2, c, r, alpha = self.unpack_params(params)

        scale = torch.tensor([float(W), float(H)], device=device, dtype=dtype)
        ref = float(min(H, W))
        P0p = P0 * scale
        P1p = P1 * scale
        P2p = P2 * scale
        r_px = r * ref                  # (B, 1) 方框半轴长（像素）

        # ---- 段1：几何（r 无关）+ stamp（几何无关）+ AABB（吃 r）+ 端点切线 ----
        bg = BezierGeom(P0p, P1p, P2p)
        s_targets, weights = StampGeom.setup(
            bg.length, r_px, self.stamp_spacing_rho, self.max_stamp_K, self.sigmoid_alpha
        )
        centers = bg.sample(s_targets, self.arc_length_grid)   # (B, K, 2)
        sg = StampGeom(centers, weights, r_px)                 # 采样集 + 半径

        # AABB 派生链：曲线 AABB + r -> tube/square/padded/integer（贴回裁剪用）
        aabb_min, aabb_max = bg.aabb
        ag = AABBGeom(aabb_min, aabb_max, r_px, self.aabb_pad)
        # 注：ix0/iy0 可越界（负 / 超 canvas），段2 paste 时裁剪到画布内。

        # 端点切线 -> CapsGeom（r 无关，仅 square）
        t0 = torch.zeros(params.size(0), 1, device=device, dtype=dtype)
        t1 = torch.ones(params.size(0), 1, device=device, dtype=dtype)
        d0 = bg.derivative(t0)[:, 0, :]                          # (B, 2) B'(0)
        d1 = bg.derivative(t1)[:, 0, :]                          # (B, 2) B'(1)
        u0 = d0 / (torch.linalg.norm(d0, dim=-1, keepdim=True) + 1e-12)
        u2 = d1 / (torch.linalg.norm(d1, dim=-1, keepdim=True) + 1e-12)
        caps = CapsGeom(u0, u2)

        # ---- 段2（快速渲染，不动）：AABB 局部渲染 + 贴回 + 颜色覆盖 ----
        # 按显存预算分 chunk，防 B·K·L² 爆炸（如大半径笔触 aabb 接近全图）
        B = params.size(0)
        L_max = max(1, int(ag.aabb_px.max().item()))
        # 截断非活跃 stamp：weight < 1e-4 的 stamp 对覆盖贡献 ≈0，
        # 可大幅缩减 K（大半径笔触 K 从 200 -> ~3-5）。
        active = sg.weights > 1e-4                             # (B, K)
        max_active = max(1, int(active.sum(dim=1).max().item()))
        if max_active < self.max_stamp_K:
            sg.weights = sg.weights[:, :max_active]
            sg.centers = sg.centers[:, :max_active, :]
            K = max_active
        else:
            K = self.max_stamp_K

        budget = 400_000_000                     # 每 chunk 中间张量元素上限
        per_pen = K * L_max * L_max
        chunk = max(1, budget // per_pen) if per_pen > 0 else B

        all_color, all_alpha = [], []
        for s in range(0, B, chunk):
            e = min(s + chunk, B)
            sl = slice(s, e)
            cov = self._forward_aabb_patch_fast(
                sg.centers[sl], sg.weights[sl], sg.r_px[sl], caps.u0[sl], caps.u2[sl],
                P0p[sl], P2p[sl], ag.aabb_px[sl], ag.ix0[sl], ag.iy0[sl], H, W,
            )  # (chunk, H, W)
            A = alpha[sl].unsqueeze(-1) * cov
            color_chunk = c[sl, :, None, None].expand(-1, -1, H, W)
            alpha_chunk = A[:, None, :, :]
            all_color.append(color_chunk)
            all_alpha.append(alpha_chunk)

        color = torch.cat(all_color, dim=0)   # (B, 3, H, W)
        alpha_out = torch.cat(all_alpha, dim=0)  # (B, 1, H, W)
        return color, alpha_out

    def _forward_aabb_patch_fast(
        self,
        centers: Tensor,      # (B, K, 2)  stamp 中心（原始像素坐标）
        weights: Tensor,      # (B, K)    软计数权重
        r_px: Tensor,         # (B, 1)    像素半径
        u0: Tensor,           # (B, 2)    起点单位切线
        u2: Tensor,           # (B, 2)    终点单位切线
        P0p: Tensor,          # (B, 2)    起点像素坐标
        P2p: Tensor,          # (B, 2)    终点像素坐标
        aabb_px: Tensor,      # (B,) long  AABB 正方形边长
        ix0: Tensor,          # (B,) long  AABB 左上 x
        iy0: Tensor,          # (B,) long  AABB 左上 y
        H: int,               # 原始 patch 高
        W: int,               # 原始 patch 宽
    ) -> Tensor:
        """段2 快速渲染 + 贴回原 patch。

        Returns:
            ``(B, H, W)`` 覆盖度（单通道），颜色逻辑由调用方处理。
        """
        device, dtype = centers.device, centers.dtype
        B = centers.size(0)

        # 1. 局部 patch 边长 = 整批 AABB 最大边长
        L = max(1, int(aabb_px.max().item()))

        # 2. 平移矩阵：geom 从原始像素坐标 → AABB 局部像素坐标
        offset = torch.stack([ix0.float(), iy0.float()], dim=-1)  # (B, 2)
        centers_l = centers - offset[:, None, :]                   # (B, K, 2)
        P0p_l = P0p - offset                                       # (B, 2)
        P2p_l = P2p - offset                                       # (B, 2)
        # r_px, weights, u0, u2 平移不变，直通

        # 3. 局部像素网格（kernel = L）
        grid_l = pixel_grid(L, L, device, dtype)  # (L, L, 2)

        # 4. 内联 render_coverage：soft-disk 并集覆盖
        diff = centers_l[:, :, None, None, :] - grid_l[None, None, :, :, :]  # (B, K, L, L, 2)
        dist = torch.linalg.norm(diff, dim=-1)                                # (B, K, L, L)
        disk = torch.sigmoid(self.disk_softness * (1.0 - dist / r_px[:, :, None, None]))  # (B, K, L, L)
        m = weights[:, :, None, None] * disk                                  # (B, K, L, L)
        cov_round = 1.0 - torch.prod(1.0 - m, dim=1)                          # (B, L, L)

        # 5. 内联 cut_endcaps：半平面 sigmoid 端点切割
        dot0 = ((grid_l[None, ...] - P0p_l[:, None, None, :]) * u0[:, None, None, :]).sum(-1)  # (B, L, L)
        cut0 = torch.sigmoid(self.endcap_sharpness * dot0)
        dot2 = ((P2p_l[:, None, None, :] - grid_l[None, ...]) * u2[:, None, None, :]).sum(-1)  # (B, L, L)
        cut2 = torch.sigmoid(self.endcap_sharpness * dot2)
        cov = cov_round * cut0 * cut2  # (B, L, L)

        # 6. 贴回原 patch：从 local 空间反 scatter（O(B·L²)，不触达 H·W 全量）
        # 对 local patch 每个像素 (ly, lx)，转置到 patch 坐标 (iy0+ly, ix0+lx)，
        # 合法则直接赋值。只遍历 L² 个小空间，patch 带宽保持在小空间内。
        ly, lx = torch.meshgrid(
            torch.arange(L, device=device),
            torch.arange(L, device=device),
            indexing='ij',
        )  # (L, L)
        ly = ly.reshape(-1)  # (L²,)
        lx = lx.reshape(-1)

        px = lx[None, :] + ix0[:, None]  # (B, L²)  patch 空间 x
        py = ly[None, :] + iy0[:, None]  # (B, L²)  patch 空间 y

        # 仅 scatter 各样本有效 AABB 区域（aabb_px 界定），其余为 padding 跳过
        in_aabb = (lx[None, :] < aabb_px[:, None]) & (ly[None, :] < aabb_px[:, None])  # (B, L²)
        in_patch = (px >= 0) & (px < W) & (py >= 0) & (py < H)                        # (B, L²)
        valid = in_aabb & in_patch

        b_idx = torch.arange(B, device=device)[:, None].expand(-1, L * L)  # (B, L²)
        flat_idx = b_idx[valid] * (H * W) + py[valid] * W + px[valid]     # (N,) 散列
        flat_vals = cov.reshape(B, -1)[valid]                               # (N,) 源值

        patch_cov = torch.zeros(B * H * W, device=device, dtype=dtype)
        patch_cov[flat_idx] = flat_vals
        return patch_cov.reshape(B, H, W)


    @classmethod
    def main(cls):
        """随机参数目视检查：生成一条随机 Bézier 方刷并显示。"""
        import matplotlib.pyplot as plt

        torch.manual_seed(0)
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 随机参数（不做约束，归一化 [0,1]）。
        params = torch.rand(1, PARAM_DIM, device=dev)
        # 让起终点拉开以得到明显曲线。
        params[0, 0:2] = torch.tensor([0.15, 0.20], device=dev)
        params[0, 4:6] = torch.tensor([0.80, 0.75], device=dev)
        params[0, 2:4] = torch.tensor([0.70, 0.15], device=dev)
        params[0, 9:10] = torch.tensor([0.05], device=dev)   # r（方框半轴长）
        params[0, 6:9] = torch.tensor([0.9, 0.2, 0.2], device=dev)  # 红
        params[0, 10:11] = torch.tensor([0.9], device=dev)   # α

        brush = cls()
        color, alpha = brush.forward(params, (256, 256))
        a = alpha[0, 0].detach().cpu().numpy()
        # 非预乘：color 是纯色常数图，笔刷形状看 alpha；blend 到白底目视
        rgb = color[0].permute(1, 2, 0).detach().cpu().numpy() * a[..., None] \
            + (1 - a[..., None])

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(rgb)
        axes[0].set_title("RGB (premultiplied)")
        axes[0].axis("off")
        axes[1].imshow(a, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Alpha")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    BezierSquareBrush.main()
