"""PixelL1Loss — 像素 L1 损失原语（无状态）。

对应 BofP ``L_app`` 中的 ``L_pixel = ‖I_r − I_t‖₁`` 项：渲染图与目标在 RGB
空间的逐像素差。零依赖、最便宜，常作 OT 的主导/兜底项（SNP 即用 L1 + L_OT 完成优化）。

原语语义（与 PointCloudOTLoss / Grad / Area 同级，统一契约）
-------------------------------------------------------------
- **无状态**：``self`` 不持 cache。``forward`` 忠实单次计算，输入→输出。
- :meth:`forward` 返回 ``(scalar, info)``：scalar 可 backward 到 pred；info 带出
  可视化所需中间量（逐像素差热图），零重算。
- :meth:`visual` —— 吃 forward 的 info 组装 viz（纯函数，不读 self）。

设计要点
--------
- **维度范式**：计算空间统一四维 ``(B,C,H,W)``。输入 pred / target 均 (1,C,H,W)。
- 参数硬编码在 ``__init__``（reduction / 平均方式），无命令行 / 配置文件。
"""

from typing import Dict, Tuple

import torch
from torch import Tensor

from .loss_base import LossBase


class PixelL1Loss(LossBase):
    """像素 L1 损失原语（无状态）。

    输入约定（按需：L1 比较色彩，只需 RGB）：
        pred   : (B,3,H,W) float，当前画布状态（带梯度，合成后的 RGB canvas）。
        target : (B,3,H,W) float，目标图。

    输出（统一契约 ``(scalar, info)``）：
        forward(...) → (标量 loss, info dict)，scalar 可 backward 到 pred。
    """

    def __init__(self):
        # ---- 死代码参数（硬编码在 init）----
        # reduction="mean"：对所有像素 + 通道取均值，量级与 OT 可比、便于加权。
        self.reduction = "mean"

    # ------------------------------------------------------------------ #
    # 常用 loss（标量，可 backward）+ info
    # ------------------------------------------------------------------ #
    def forward(self, pred: Tensor, target: Tensor) -> Tuple[Tensor, dict]:
        """计算 pred 与 target 的 L1 距离。

        Args:
            pred: (B,3,H,W) 当前画布状态，带梯度。
            target: (B,3,H,W) 目标图（内部 detach）。

        Returns:
            (loss, info)：loss 标量；info 带出逐像素 L1 差 (B,3,H,W)（detach），
            供 :meth:`visual` 零重算组装 viz。
        """
        pred = pred.float()
        tgt = target.detach().float()
        diff = (pred - tgt).abs()                        # (1,C,H,W)，带梯度到 pred

        if self.reduction == "mean":
            loss = diff.mean()
        elif self.reduction == "sum":
            loss = diff.sum()
        else:
            raise ValueError(f"未知 reduction: {self.reduction}")

        info = {
            "diff": diff.detach(),   # (1,C,H,W) 逐像素 L1 差
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
                - "l1_diff" : 逐像素 L1 差热图（按通道均值压成 (B,1,H,W) 便于 imshow）
        """
        diff = info["diff"].detach()                     # (1,C,H,W)
        return {
            "l1_diff": diff.mean(1, keepdim=True),       # (1,1,H,W)
        }

    # ------------------------------------------------------------------ #
    # 设计注记
    # ------------------------------------------------------------------ #
    # 原语无状态：cache 全部上移到 loss_space。本类只忠实算一次 (pred,target)→loss。
