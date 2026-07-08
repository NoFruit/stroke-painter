"""AreaLoss — 笔触面积正则损失原语（无状态）。

对应 BofP ``L_app`` 中的 ``L_area`` 项（Optimization Regularization）：惩罚面积趋于
0 的退化笔触，确保每笔维持最小有效足迹。BofP 原式 ``exp(−area(A_s)/η)`` 对每笔
stroke mask 取面积；本项目以"渲染图的覆盖度 alpha"作为笔触面积代理——

    L_area = exp(− mean(A) / η)

其中 ``A`` 为渲染笔刷的 alpha 覆盖图（笔触落笔越多，mean(A) 越大 → loss 越小）。
笔触退化（alpha→0）时 loss→1，强惩罚；正常落笔时 loss→0。

**输入约定差异**：本原语需要**笔刷的 alpha 覆盖图**，而非合成后的 RGB canvas。
   故 forward 接收 ``pred_alpha`` (B,1,H,W)（mainloop 持有笔刷 forward 的 alpha 输出，带梯度），
   ``target`` 在此原语中**不使用**（仅占位以统一 forward 签名）。

原语语义（与 PointCloudOTLoss / L1 / Grad 同级，统一契约）
-------------------------------------------------------------
- **无状态**：``self`` 不持 cache。``forward`` 忠实单次计算，输入→输出。
- :meth:`forward` 返回 ``(scalar, info)``：scalar 可 backward 到 pred_alpha；info 带
  出可视化所需中间量（alpha 覆盖图），零重算。
- :meth:`visual` —— 吃 forward 的 info 组装 viz（纯函数，不读 self）。

设计要点
--------
- **维度范式**：计算空间统一四维。``pred_alpha`` 为 (B,1,H,W)（单通道覆盖度）。
- 参数硬编码在 ``__init__``（η），无命令行 / 配置文件。
- forward 签名 ``forward(pred, target)`` 与其它原语统一；target 在此忽略（占位），
  由 loss_space 传入（loss_space 本就持有 target，统一调用形态更整齐）。
"""

from typing import Dict, Tuple

import torch
from torch import Tensor

from .loss_base import LossBase


class AreaLoss(LossBase):
    """笔触面积正则损失原语（无状态）。

    输入约定（按需：Area 只需 alpha 覆盖度，不需色彩）：
        pred   : (B,1,H,W) float，笔刷 alpha 覆盖图（带梯度，mainloop 传笔刷 forward 的 alpha 输出）。
        target : 任意（本原语忽略，仅占位以统一 forward 签名）。

    输出（统一契约 ``(scalar, info)``）：
        forward(...) → (标量 loss, info dict)，scalar 可 backward 到 pred。
    """

    def __init__(self):
        # ---- 死代码参数（硬编码在 init）----
        self.eta = 0.05          # 面积尺度 η：mean(A)≈η 时 loss≈e⁻¹；越小越严苛
        self.eps = 1e-8

    # ------------------------------------------------------------------ #
    # 常用 loss（标量，可 backward）+ info
    # ------------------------------------------------------------------ #
    def forward(self, pred: Tensor, target: Tensor) -> Tuple[Tensor, dict]:
        """计算笔触面积正则损失。

        Args:
            pred: (B,1,H,W) 笔刷 alpha 覆盖图，带梯度。
            target: 忽略（占位）。

        Returns:
            (loss, info)：loss 标量（退化→1，正常落笔→0）；info 带出 alpha 覆盖图
            (B,1,H,W)（detach），供 :meth:`visual` 零重算组装 viz。
        """
        a = pred.float()                                  # (B,1,H,W) 带梯度
        area = a.mean()                                   # 标量，带梯度
        loss = torch.exp(-area / (self.eta + self.eps))   # 退化→1，落笔→0

        info = {
            "alpha": a.detach(),                          # (B,1,H,W)
            "area": float(area.detach()),
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
                - "alpha" : 笔刷 alpha 覆盖图（笔触足迹）
        """
        return {
            "alpha": info["alpha"].detach(),              # (B,1,H,W)
        }

    # ------------------------------------------------------------------ #
    # 设计注记
    # ------------------------------------------------------------------ #
    # 原语无状态：cache 全部上移到 loss_space。本类只忠实算一次 (pred_alpha,*)→loss。
    # 输入是 alpha 覆盖图而非 RGB canvas--由 mainloop 从笔刷 forward 的 alpha 输出传入。
