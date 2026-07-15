"""LineGeom - 直线的纯几何生产者（r 无关）。

与 :class:`BezierGeom` 平行：把直线的 1D 解析式（长度 / 方向 / 曲线 AABB /
弧长采样）包成缓存对象。直线无内部极值，AABB 即端点 min/max；弧长反演是闭式
``t = s/L``，无需网格 M。
"""

import torch
from torch import Tensor


class LineGeom:
    """直线 1D 解析式缓存（r 无关、空间无关）。

    Args:
        p0, p2: 端点，形状 ``(B, 2)``。

    Attributes:
        p0, p2: 端点（原样保存），各 ``(B, 2)``。
        length: ``‖P2 - P0‖``，``(B,)``（解析）。
        direction: 单位方向 u = normalize(P2 - P0)，``(B, 2)``（直线两端切线
            相同，供 :class:`CapsGeom` 用）。
    """

    def __init__(self, p0: Tensor, p2: Tensor) -> None:
        self.p0 = p0
        self.p2 = p2
        diff = p2 - p0                                     # (B,2)
        self.length = torch.linalg.norm(diff, dim=-1)      # (B,)
        self.direction = diff / (self.length[..., None] + 1e-12)  # (B,2)
        self._aabb: tuple[Tensor, Tensor] | None = None

    def sample(self, s_targets: Tensor) -> Tensor:
        """直线弧长采样：``t = (s/L).clamp(max=1)``，``centers = (1-t)P0 + t P2``。

        无 M（直线弧长反演是闭式 s/L）。超出 [0,L] 的目标映射到终点 t=1
        （其 stamp 权重由软计数决定，通常 ~0）。

        Args:
            s_targets: ``(B, K)`` 目标弧长。
        Returns:
            ``(B, K, 2)`` stamp 中心。
        """
        t = (s_targets / (self.length[:, None] + 1e-12)).clamp(max=1.0)  # (B,K)
        return (
            (1.0 - t)[..., None] * self.p0[:, None, :]
            + t[..., None] * self.p2[:, None, :]
        )  # (B,K,2)

    @property
    def aabb(self) -> tuple[Tensor, Tensor]:
        """曲线 AABB ``(aabb_min, aabb_max)``，各 ``(B, 2)``。

        直线无内部极值，AABB 即两端点分量 min/max。r 无关；膨胀归 :class:`AABBGeom`。
        """
        if self._aabb is None:
            pts = torch.stack([self.p0, self.p2], dim=1)      # (B,2,2)
            self._aabb = (pts.amin(dim=1), pts.amax(dim=1))  # 各 (B,2)
        return self._aabb
