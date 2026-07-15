"""StampGeom - 绘制前夕的几何无关 stamp 数据（采样集 + 半径）。

无论 line / bezier / 其他曲线，到 :func:`render_coverage` 绘制前夕都归约成同一
份参数：采样集（centers + 软计数权重 weights）+ 像素半径 r_px。本类只持有这
份参数，**不知曲线类型**；曲线专属的采样（弧长 -> 点）由曲线 geom 的
``.sample`` 产出后传入。

通用 stamp 参数设置（d / s_targets / weights，由曲线长度 L + 半径 r_px 产出）
作为 :meth:`StampGeom.setup` classmethod 提供，避免在各笔刷重复。
"""

import torch
from torch import Tensor

from ..utils.stamp import soft_stamp_count


class StampGeom:
    """几何无关的 stamp 持有体 = 采样集 + 半径。

    Args:
        centers: ``(B, K, 2)`` stamp 中心（由曲线 geom 的 ``.sample`` 产出）。
        weights: ``(B, K)`` 软计数激活权重（采样集的一部分）。
        r_px: ``(B, 1)`` 像素半径。

    Attributes:
        centers, weights, r_px: 同上（即 :func:`render_coverage` 的前三个输入）。

    Note:
        不持有 ``grid_coords`` / ``disk_softness`` -- 那些是 ``render_coverage``
        的绘制侧输入，调用时传。
    """

    def __init__(self, centers: Tensor, weights: Tensor, r_px: Tensor) -> None:
        self.centers = centers   # (B, K, 2)
        self.weights = weights   # (B, K)
        self.r_px = r_px          # (B, 1)

    @classmethod
    def setup(
        cls,
        L: Tensor,
        r_px: Tensor,
        rho: float,
        K: int,
        alpha: float,
    ) -> tuple[Tensor, Tensor]:
        """通用 stamp 参数设置（几何无关）：L + r_px -> (s_targets, weights)。

        ``d = r_px * rho``（stamp 间隔）、``s_targets = d * k``（k=1..K 目标弧长）、
        ``weights = soft_stamp_count(L, d, K, alpha)``（软计数激活）。centers 由
        曲线 geom 的 ``.sample(s_targets, M)`` 另行产出（曲线专属），再与本
        classmethod 的返回值一起构造 :class:`StampGeom`。

        Args:
            L: ``(B,)`` 曲线长度（来自曲线 geom 的 ``.length``）。
            r_px: ``(B, 1)`` 像素半径。
            rho: stamp 间隔比例（``d = r_px * rho``）。
            K: 候选 stamp 数上界。
            alpha: 软计数 sigmoid 陡度。

        Returns:
            (s_targets, weights)：``s_targets`` ``(B, K)``、``weights`` ``(B, K)``。
        """
        d = r_px * rho                                              # (B,1) stamp 间隔
        k = torch.arange(1, K + 1, device=L.device, dtype=L.dtype)  # (K,)
        s_targets = d * k[None, :]                                  # (B,K)
        weights = soft_stamp_count(L, d, K, alpha)                  # (B,K)
        return s_targets, weights
