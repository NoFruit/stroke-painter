"""BezierGeom - 二次 Bézier 曲线的纯几何生产者（r 无关）。

把 :mod:`diffbrush.utils.bezier` 的纯函数包成「构造时算一次、缓存到 self」的
对象，供绘制管线上游复用。本类只负责 1D 解析式（控制点 -> 系数 / 弧长 / 极值 /
曲线 AABB / 弧长采样），**不碰半径 r、不碰绘制**（``render_coverage`` 等）。

空间无关：构造时给什么空间的控制点，所有几何量就在什么空间（笔刷传入像素坐标
``P0p`` 则量在像素空间）。

二次 Bézier ``B(t) = (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2 = a t^2 + b t + c``，
其中 ``a = P0 - 2 P1 + P2``、``b = 2(P1 - P0)``、``c = P0``。
"""

import torch
from torch import Tensor

from ..utils.bezier import (
    arc_length_to_t,
    bezier_coefficients,
    bezier_derivative,
    bezier_eval,
    bezier_extrema,
    bezier_length,
)


class BezierGeom:
    """二次 Bézier 1D 解析式缓存（r 无关、空间无关）。

    构造时 eager 计算系数 (a,b,c) 与闭式弧长 L；极值 / 曲线 AABB 懒加载
    （仅需要 AABB 的快速路径才付代价，全量 forward 不触发）。

    Args:
        p0, p1, p2: 控制点，形状 ``(B, 2)``。

    Attributes:
        p0, p1, p2: 控制点（原样保存），各 ``(B, 2)``。
        a, b, c: 解析式系数，各 ``(B, 2)``（``bezier_coefficients``）。
        length: 弧长 L，``(B,)``（``bezier_length`` 闭式解）。
    """

    def __init__(self, p0: Tensor, p1: Tensor, p2: Tensor) -> None:
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        # 控制点 -> 解析式系数（线性、可微；算一次复用，省去重复展开）。
        self.a, self.b, self.c = bezier_coefficients(p0, p1, p2)  # 各 (B,2)
        # 闭式弧长（c 不参与，弧长平移不变）。
        self.length = bezier_length(self.a, self.b, self.c)       # (B,)
        # 懒加载缓存（仅 forward_fast 等 AABB 路径触发）。
        self._extrema: Tensor | None = None
        self._aabb: tuple[Tensor, Tensor] | None = None

    # ------------------------------------------------------------------ eager

    @property
    def abc(self) -> tuple[Tensor, Tensor, Tensor]:
        """解析式系数 ``(a, b, c)``，各 ``(B, 2)``。"""
        return self.a, self.b, self.c

    def eval(self, t: Tensor) -> Tensor:
        """B(t) = a t^2 + b t + c。

        Args:
            t: ``(B, K)`` 曲线参数。
        Returns:
            ``(B, K, 2)`` 曲线点。
        """
        return bezier_eval(self.a, self.b, self.c, t)

    def derivative(self, t: Tensor) -> Tensor:
        """B'(t) = 2 a t + b（未归一化）。

        Args:
            t: ``(B, K)``。
        Returns:
            ``(B, K, 2)`` 导数向量。
        """
        return bezier_derivative(self.a, self.b, self.c, t)

    def sample(self, s_targets: Tensor, M: int) -> Tensor:
        """弧长参数化采样：给目标弧长，返回曲线上对应的点。

        内部 = ``arc_length_to_t(a,b,c, s, M)`` -> ``bezier_eval(a,b,c, t)``。
        r 无关（只吃 s_targets）；``M`` 为弧长反演网格分辨率（传入，构造器保持
        纯几何）。超出 [0, L] 的目标映射到端点 t=1（其 stamp 权重由软计数决定）。

        Args:
            s_targets: ``(B, K)`` 目标弧长。
            M: 弧长反演网格分辨率（``arc_length_to_t`` 的 M）。
        Returns:
            ``(B, K, 2)`` stamp 中心（曲线点）。
        """
        t = arc_length_to_t(self.a, self.b, self.c, s_targets, M)  # (B,K)
        return bezier_eval(self.a, self.b, self.c, t)              # (B,K,2)

    # ------------------------------------------------------------------ lazy

    @property
    def extrema(self) -> Tensor:
        """极值点参数 ``(B, 2) = [t_x, t_y]``，已 clamp 到 [0,1]（懒加载）。"""
        if self._extrema is None:
            self._extrema = bezier_extrema(self.a, self.b, self.c)  # (B,2)
        return self._extrema

    @property
    def aabb(self) -> tuple[Tensor, Tensor]:
        """曲线 AABB ``(aabb_min, aabb_max)``，各 ``(B, 2)``（懒加载，r 无关）。

        候选 t = {0, 1, t_x, t_y}（极值已 clamp 到 [0,1]，超范围临界点塌缩到
        端点，已在候选集中无害）求值后分量 min/max，解析精确。``r`` 的膨胀（tube）
        归 :class:`AABBGeom`，不在本类。
        """
        if self._aabb is None:
            B = self.a.shape[0]
            device, dtype = self.a.device, self.a.dtype
            t_ext = self.extrema                                     # (B,2)
            t_cand = torch.stack([
                torch.zeros(B, device=device, dtype=dtype),
                torch.ones(B, device=device, dtype=dtype),
                t_ext[:, 0],
                t_ext[:, 1],
            ], dim=1)                                                # (B,4) 候选 t
            cand_pts = bezier_eval(self.a, self.b, self.c, t_cand)  # (B,4,2)
            self._aabb = (cand_pts.amin(dim=1), cand_pts.amax(dim=1))  # 各 (B,2)
        return self._aabb
