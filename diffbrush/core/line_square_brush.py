"""LineSquareBrush — 单色、均匀粗细的直线方头笔刷（disk-SDF + 端点切割）。

与 :class:`UniformLineBrush` 共享 disk-SDF 圆刷覆盖，然后在两端点施加半平面
sigmoid 切割，将半圆头切为方头。直线时两端切线相同（u0 = u2 = 直线方向），
两个切平面平行。原理::

    M_round(x) = 1 − ∏_k(1 − w_k·disk_k)         # 圆刷覆盖（同 UniformLineBrush）
    M_square(x) = M_round(x) · cut_start(x) · cut_end(x)   # 端点切割

参数（归一化输入，flat 向量，共 9 维）::

    Θ = { P0, P2, c, r, α }

    [0:2]   P0  起点坐标      (2,)   归一化 [0,1] → 像素 [0,W]×[0,H]
    [2:4]   P2  终点坐标      (2,)   同上
    [4:7]   c   单色 RGB      (3,)   [0,1]
    [7:8]   r   均匀半径      (1,)   归一化 → 像素 r·min(H,W)
    [8:9]   α   透明度        (1,)   [0,1]

渲染流程（可微）：
  1. 端点 / 半径归一化 → 像素坐标。
  2. 直线长度 L = ||P2 − P0||（解析）；间隔 d = r·ρ。
  3. Sigmoid 软计数：w_k = σ(α_cnt·(L/d − k))，k=1..K。
  4. stamp k 置于弧长 s_k = k·d 处（直线 t = s/L）。
  5. soft-disk 并集覆盖 M_round(x) = 1 − ∏_k(1 − w_k·disk_k)。
  6. 端点半平面切割：M = M_round · cut_endcaps(...)。
  7. 输出 (color, alpha) 双通道：color = c 广播 (B,3,H,W)（非预乘纯色），
     alpha = α·M (B,1,H,W)（覆盖度）。
"""
from typing import Tuple

import torch
from torch import Tensor

from .brush_base import BrushBase, ParamLayout
from ..utils.stamp import pixel_grid, cut_endcaps, render_coverage, soft_stamp_count

PARAM_DIM = 9


class LineSquareBrush(BrushBase):
    @classmethod
    def param_layout(cls) -> ParamLayout:
        """Line 笔刷 (9-dim) 语义通道布局：P0[0:2], P2[2:4], c[4:7], r[7], α[8]。"""
        return ParamLayout(
            param_dim=PARAM_DIM,
            point_slices=((0, 2), (2, 4)),
            color_slice=(4, 7),
            radius_idx=7,
            alpha_idx=8,
        )

    @staticmethod
    def unpack_params(params: Tensor):
        """(B, 9) → P0, P2, c, r, alpha，各形状 (B, *)。"""
        P0 = params[:, 0:2]
        P2 = params[:, 2:4]
        c = params[:, 4:7]
        r = params[:, 7:8]
        alpha = params[:, 8:9]
        return P0, P2, c, r, alpha

    def __init__(self):
        # 固定数值超参（design_paradigm：作为 self 变量硬编码，不外置、不进 config）。
        self.sigmoid_alpha = 100.0       # 软计数 sigmoid 陡度
        self.max_stamp_K = 100          # 候选 stamp 数上界
        self.disk_softness = 100.0       # soft-disk 边缘陡度（无量纲，相对半径）
        self.stamp_spacing_rho = 0.5    # stamp 间隔比例 ρ：d = r·ρ（ρ<1 → 重叠覆盖）
        self.endcap_sharpness = 100.0   # 端点切割 sigmoid 陡度 β

    def forward(self, params: Tensor, patch_size) -> Tuple[Tensor, Tensor]:
        """归一化参数 → (color, alpha)。

        Args:
            params: ``(B, 9)`` 或 ``(9,)`` 归一化参数（见模块 docstring）。
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

        P0, P2, c, r, alpha = self.unpack_params(params)

        scale = torch.tensor([float(W), float(H)], device=device, dtype=dtype)
        ref = float(min(H, W))
        P0p = P0 * scale
        P2p = P2 * scale
        r_px = r * ref                  # (B, 1) 方框半轴长（像素）
        d = r_px * self.stamp_spacing_rho  # (B, 1) stamp 间隔（像素）

        # 直线长度（解析，无需数值积分）。
        L = torch.linalg.norm(P2p - P0p, dim=-1)  # (B,)
        weights = soft_stamp_count(L, d, self.max_stamp_K, self.sigmoid_alpha)  # (B, K)

        k = torch.arange(1, self.max_stamp_K + 1, device=device, dtype=dtype)
        s_targets = d * k[None, :]      # (B, K) 每个 stamp 的目标弧长 s_k = k·d
        # 直线弧长参数化：t = s / L；超出 [0,L] 的 stamp 映射到终点 t=1（权重由软计数决定）。
        t_k = (s_targets / (L[:, None] + 1e-12)).clamp(max=1.0)  # (B, K)
        centers = (1.0 - t_k)[..., None] * P0p[:, None, :] + t_k[..., None] * P2p[:, None, :]  # (B, K, 2)

        grid_coords = pixel_grid(H, W, device, dtype)  # (H, W, 2)
        coverage_round = render_coverage(
            centers, r_px, weights, grid_coords, self.disk_softness
        )  # (B, H, W)

        # 直线方向即两端切线 u0 = u2 = normalize(P2 − P0)
        direction = P2p - P0p  # (B, 2)
        u = direction / (torch.linalg.norm(direction, dim=-1, keepdim=True) + 1e-12)  # (B, 2)

        coverage = cut_endcaps(
            coverage_round, grid_coords, P0p, P2p, u, u, self.endcap_sharpness
        )  # (B, H, W)

        A = alpha.unsqueeze(-1) * coverage                # (B, H, W) 覆盖度 = α·M
        color = c[:, :, None, None].expand(-1, -1, H, W)  # (B, 3, H, W) 纯色（非预乘）
        alpha_out = A[:, None, :, :]                      # (B, 1, H, W) 覆盖度
        return color, alpha_out

    @classmethod
    def main(cls):
        """随机参数目视检查：生成一条随机直线方刷并显示。"""
        import matplotlib.pyplot as plt

        torch.manual_seed(0)
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 随机参数（不做约束，归一化 [0,1]）。
        params = torch.rand(1, PARAM_DIM, device=dev)
        # 让起终点拉开以得到明显直线。
        params[0, 0:2] = torch.tensor([0.15, 0.20], device=dev)
        params[0, 2:4] = torch.tensor([0.85, 0.80], device=dev)
        params[0, 7:8] = torch.tensor([0.05], device=dev)   # r（方框半轴长）
        params[0, 4:7] = torch.tensor([0.9, 0.2, 0.2], device=dev)  # 红
        params[0, 8:9] = torch.tensor([0.9], device=dev)   # α

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
    LineSquareBrush.main()
