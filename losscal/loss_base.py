"""LossBase — 损失计算器抽象基类契约。

所有损失原语实现必须继承本类并实现 :meth:`forward` 与 :meth:`visual`，遵循
"无状态 ``forward→(scalar, info)`` + ``visual(info)→dict``" 的统一契约：

- :meth:`forward` 忠实单次计算，返回 ``(scalar, info)``：scalar 为可经 autograd
  反传至输入张量（渲染图 / 质量分布）的标量损失；info 为带出可视化所需中间量的
  dict（零重算）。
- :meth:`visual` 是纯函数：吃一次 forward 的 info，组装成 viz 字典（不读 self 状态）。

原语无状态：``self`` 不持 cache、不缓存 target。组合 / 加权 / target 跨步复用
由上层（loss_space 等）统一持有，单一 cache owner。
"""

from abc import ABC, abstractmethod
from typing import Tuple

from torch import Tensor


class LossBase(ABC):
    @abstractmethod
    def forward(self, pred: Tensor, target: Tensor, *args, **kwargs) -> Tuple[Tensor, dict]:
        """预测分布 → (标量损失, info)。

        Args:
            pred: 预测分布 / 渲染图。**形状与通道由子类按自身需要定义**
                （如 L1/Grad/OT 吃 RGB (B,3,H,W)，Area 吃 alpha (B,1,H,W)），
                去掉冗余通道--不强制统一格式。
            target: 目标分布 / 目标图，形状与 pred 对应（Area 等单输入原语忽略）。
            *args, **kwargs: 实现类自定义的额外输入。

        Returns:
            (loss, info)：loss 为标量损失张量，梯度可通到 ``pred``；
            info 为 dict，带出 :meth:`visual` 所需中间量（已 detach）。
        """
        ...

    def __call__(self, *args, **kwargs) -> Tuple[Tensor, dict]:
        """调用即 forward（轻量转发，不引入 nn.Module 开销）。"""
        return self.forward(*args, **kwargs)

    @abstractmethod
    def visual(self, info: dict) -> dict:
        """把一次 :meth:`forward` 的 info 整理成 viz 字典（纯函数，不读 self）。

        Args:
            info: 一次 :meth:`forward` 返回的 info dict。

        Returns:
            viz dict，键值通常为已 detach 的张量（四维标准 (B, C, H, W)），
            由上层 viewer 导出 / 降维展示。
        """
        ...
