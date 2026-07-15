"""CapsGeom - 方头切割的几何量 u0/u2（r 无关，仅 square）。

端点切线 u0（起点）/ u2（终点），供 :func:`cut_endcaps` 半平面 sigmoid 切割用。
只有方头笔刷实例化（圆头 uniform 不需要，取代 None-跳过范式）。r 无关（纯切线）；
``sharpness`` 是绘制侧超参，调用时传。

本类只持有已归一化的单位切线，**不依赖任何曲线 geom 类**（保持解耦）；切线的计算
（bezier: ``B'(0)`` / ``B'(1)`` 归一化；line: ``direction``）由调用方用曲线 geom 的
``.derivative`` / ``.direction`` 产出后传入。
"""

from torch import Tensor


class CapsGeom:
    """方头端点切线持有体（r 无关）。

    Args:
        u0: ``(B, 2)`` 起点处单位切线（方向指向曲线内部）。
        u2: ``(B, 2)`` 终点处单位切线（方向指向曲线内部）。

    Attributes:
        u0, u2: 同上。
    """

    def __init__(self, u0: Tensor, u2: Tensor) -> None:
        self.u0 = u0   # (B, 2)
        self.u2 = u2   # (B, 2)
