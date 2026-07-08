"""GradientLoss — 梯度对齐损失原语（无状态）。

对应 BofP ``L_app`` 中的 ``L_grad`` 项（Structural Guidance）：用图像梯度引导笔触
沿局部几何（边缘/轮廓）走。BofP 原式沿笔触弧长平均 ``α·L_mag + β·L_dir``（幅值 +
方向）；本原语取其**图像级**等价形式——对 pred/target 各算 Sobel 梯度，惩罚梯度
幅值差与方向差，作用在整图上（与"渲染 patch → 整图 loss"流程对齐）。

具体形式：
    L_mag = mean | |∇I_r| − |∇I_t| |          （梯度幅值差）
    L_dir = mean (1 − cos)                     （梯度方向夹角，1−cos∈[0,2]）
    L_grad = α·L_mag + β·L_dir

零依赖（仅 Sobel 卷积，无可学习参数）。

原语语义（与 PointCloudOTLoss / L1 / Area 同级，统一契约）
-------------------------------------------------------------
- **无状态**：``self`` 不持 cache。``forward`` 忠实单次计算，输入→输出。
- :meth:`forward` 返回 ``(scalar, info)``：scalar 可 backward 到 pred；info 带出
  可视化所需中间量（pred/target 梯度幅值图），零重算。
- :meth:`visual` —— 吃 forward 的 info 组装 viz（纯函数，不读 self）。

设计要点
--------
- **维度范式**：计算空间统一四维 ``(B,C,H,W)``。输入 pred / target 均 (B,3,H,W)；
  内部按亮度转单通道 (B,1,H,W) 再算梯度（方向在标量场上有定义）。
- 参数硬编码在 ``__init__``（α/β），无命令行 / 配置文件。
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .loss_base import LossBase


def _sobel_xy(img: Tensor) -> Tuple[Tensor, Tensor]:
    """(B,1,H,W) → (gx, gy)，Sobel 梯度。核在 img.device 上构造，detach。

    Sobel 1/8 归一，使 |∇I| 量级与像素差可比。
    """
    dev = img.device
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=img.dtype, device=dev) / 8.0
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                      dtype=img.dtype, device=dev) / 8.0
    kx = kx.view(1, 1, 3, 3)
    ky = ky.view(1, 1, 3, 3)
    gx = F.conv2d(img, kx, padding=1)
    gy = F.conv2d(img, ky, padding=1)
    return gx, gy


class GradientLoss(LossBase):
    """梯度对齐损失原语（无状态）。

    输入约定（计算空间统一四维，默认即四维，不做检查）：
        pred   : (B,3,H,W) float [0,1]，当前画布状态（带梯度）。
        target : (B,3,H,W) float [0,1]，目标图。

    输出（统一契约 ``(scalar, info)``）：
        forward(...) → (标量 loss, info dict)，scalar 可 backward 到 pred。
    """

    def __init__(self):
        # ---- 死代码参数（硬编码在 init）----
        self.alpha = 1.0          # L_mag 权重
        self.beta = 1.0           # L_dir 权重
        self.eps = 1e-8           # 方向夹角数值稳定（防 |∇|=0 处除零）
        # 亮度系数（与 OT to_mass 的 luminance 同约定）
        self._lum = (0.299, 0.587, 0.114)

    def _to_lum(self, img: Tensor) -> Tensor:
        """(B,3,H,W) → (B,1,H,W) 亮度（标量场，方向有定义）。"""
        r, g, b = self._lum
        return r * img[:, 0] + g * img[:, 1] + b * img[:, 2]   # (B,H,W)
        # 注：返回 (B,H,W)；下面 _sobel_xy 需要 (B,1,H,W)，故加通道维

    # ------------------------------------------------------------------ #
    # 常用 loss（标量，可 backward）+ info
    # ------------------------------------------------------------------ #
    def forward(self, pred: Tensor, target: Tensor) -> Tuple[Tensor, dict]:
        """计算 pred 与 target 的梯度对齐损失（幅值差 + 方向差）。

        Args:
            pred: (B,3,H,W) 当前画布状态，带梯度。
            target: (B,3,H,W) 目标图（内部 detach）。

        Returns:
            (loss, info)：loss 标量；info 带出 pred/target 梯度幅值图 (B,1,H,W)
            （detach），供 :meth:`visual` 零重算组装 viz。
        """
        pred_l = self._to_lum(pred.float())[:, None]            # (B,1,H,W) 带梯度
        tgt_l = self._to_lum(target.detach().float())[:, None]  # (B,1,H,W) detach

        gx_r, gy_r = _sobel_xy(pred_l)                          # 带梯度
        gx_t, gy_t = _sobel_xy(tgt_l)                          # detach

        mag_r = torch.sqrt(gx_r ** 2 + gy_r ** 2 + self.eps)    # 带梯度
        mag_t = torch.sqrt(gx_t ** 2 + gy_t ** 2 + self.eps)    # detach

        # 幅值差
        l_mag = (mag_r - mag_t).abs().mean()

        # 方向差：1 − cos(θ_r, θ_t) = 1 − (g_r·g_t)/(|g_r||g_t|)
        dot = gx_r * gx_t + gy_r * gy_t
        cos = dot / (mag_r * mag_t + self.eps)
        l_dir = (1.0 - cos).mean()

        loss = self.alpha * l_mag + self.beta * l_dir

        info = {
            "pred_grad_mag": mag_r.detach(),   # (B,1,H,W)
            "target_grad_mag": mag_t.detach(), # (B,1,H,W)
        }
        return loss, info

    # ------------------------------------------------------------------ #
    # 可视组装（纯函数：吃 forward 的 info，不读 self 状态）
    # ------------------------------------------------------------------ #
    def visual(self, info: dict) -> Dict[str, Tensor]:
        """把一次 :meth:`forward` 的 info 整理成 viz 字典（纯函数，不读 self）。

        Returns:
            dict，键值均为 (B,1,H,W) 张量（已 detach，四维标准；由 viewer 导出边界
            降维展示）：
                - "pred_grad_mag"   : pred 梯度幅值图（边缘在哪）
                - "target_grad_mag" : target 梯度幅值图
                - "grad_diff"       : 幅值差，主可视——正=pred 边缘过量，负=该补边缘
        """
        mr = info["pred_grad_mag"].detach()                     # (B,1,H,W)
        mt = info["target_grad_mag"].detach()                   # (B,1,H,W)
        return {
            "pred_grad_mag": mr,
            "target_grad_mag": mt,
            "grad_diff": mr - mt,                               # 主可视
        }

    # ------------------------------------------------------------------ #
    # 设计注记
    # ------------------------------------------------------------------ #
    # 原语无状态：cache 全部上移到 loss_space。本类只忠实算一次 (pred,target)→loss。
