"""BezierUniformBrush — 单色、均匀粗细的二次 Bézier 笔刷。

继承自 :class:`BrushBase`。与 Birth-of-Painting 论文笔刷的差异：
单色（非渐变）、头尾粗细相同（均匀）、三点二次 Bézier（起点 / 控制点 / 终点）。

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
  5. soft-disk 并集覆盖 M(x) = 1 − ∏_k(1 − w_k·disk_k)。
  6. 输出 (color, alpha) 双通道：color = c 广播 (B,3,H,W)（非预乘纯色），
     alpha = α·M (B,1,H,W)（覆盖度）。
"""

from typing import Tuple

import torch
from torch import Tensor

from .brush_base import BrushBase, ParamLayout
from ..utils.bezier import (
    arc_length_to_t,
    bezier_coefficients,
    bezier_eval,
    bezier_length,
)
from ..utils.stamp import pixel_grid, render_coverage, soft_stamp_count

PARAM_DIM = 11


class BezierUniformBrush(BrushBase):
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
        self.sigmoid_alpha = 100.0  # 软计数 sigmoid 陡度
        self.max_stamp_K = 100  # 候选 stamp 数上界
        self.disk_softness = 100.0  # soft-disk 边缘陡度（无量纲，相对半径）
        self.stamp_spacing_rho = 0.5  # stamp 间隔比例 ρ：d = r·ρ（ρ<1 → 重叠覆盖）
        self.arc_length_grid = 256  # 弧长反演（arc_length_to_t 的 M）网格分辨率

    def forward(self, params: Tensor, patch_size) -> Tuple[Tensor, Tensor]:
        """归一化参数 → (color, alpha)。

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
        # 控制点 -> 解析式系数 B(t)=a_coef·t²+b_coef·t+c_coef（线性、可微；算一次复用，省去重复展开）。
        # 系数加 _coef 后缀以区别于颜色变量 c（RGB）：c_coef 即常数项 = P0p。
        a_coef, b_coef, c_coef = bezier_coefficients(P0p, P1p, P2p)
        r_px = r * ref  # (B, 1)
        d = r_px * self.stamp_spacing_rho  # (B, 1) stamp 间隔（像素）

        L = bezier_length(a_coef, b_coef, c_coef)  # (B,) 闭式弧长
        weights = soft_stamp_count(L, d, self.max_stamp_K, self.sigmoid_alpha)  # (B, K)

        k = torch.arange(1, self.max_stamp_K + 1, device=device, dtype=dtype)
        s_targets = d * k[None, :]  # (B, K) 每个 stamp 的目标弧长 s_k = k·d
        t_k = arc_length_to_t(
            a_coef, b_coef, c_coef, s_targets, self.arc_length_grid
        )  # (B, K)
        centers = bezier_eval(a_coef, b_coef, c_coef, t_k)  # (B, K, 2)

        grid_coords = pixel_grid(H, W, device, dtype)  # (H, W, 2)
        coverage = render_coverage(
            centers, r_px, weights, grid_coords, self.disk_softness
        )  # (B, H, W)

        A = alpha.unsqueeze(-1) * coverage  # (B, H, W) 覆盖度 = α·M
        color = c[:, :, None, None].expand(-1, -1, H, W)  # (B, 3, H, W) 纯色（非预乘）
        alpha_out = A[:, None, :, :]  # (B, 1, H, W) 覆盖度
        return color, alpha_out

    def forward_fast(self):
        raise NotImplementedError

    @classmethod
    def main(cls):
        """随机参数目视检查：生成一条随机 Bézier 笔刷并显示。"""
        import matplotlib.pyplot as plt

        torch.manual_seed(0)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 随机参数（不做约束，归一化 [0,1]）。
        params = torch.rand(1, PARAM_DIM, device=dev)
        # 让起终点拉开以得到明显曲线。
        params[0, 0:2] = torch.tensor([0.15, 0.20], device=dev)
        params[0, 4:6] = torch.tensor([0.80, 0.75], device=dev)
        params[0, 2:4] = torch.tensor([0.70, 0.15], device=dev)
        params[0, 9:10] = torch.tensor([0.05], device=dev)  # r
        params[0, 6:9] = torch.tensor([0.9, 0.2, 0.2], device=dev)  # 红
        params[0, 10:11] = torch.tensor([0.9], device=dev)  # α

        brush = cls()
        color, alpha = brush.forward(params, (256, 256))
        a = alpha[0, 0].detach().cpu().numpy()
        # 非预乘：color 是纯色常数图，笔刷形状看 alpha；blend 到白底目视
        rgb = color[0].permute(1, 2, 0).detach().cpu().numpy() * a[..., None] + (
            1 - a[..., None]
        )

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
    BezierUniformBrush.main()
