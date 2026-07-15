"""AABBGeom - 曲线 AABB + r -> 渲染框派生链（吃 r）。

吃曲线 geom 产出的曲线 AABB（r 无关）+ 像素半径 r_px，派生出绘制快速路径要的
各级渲染框。派生链各级独立缓存（互不覆写）：

    tube    = 曲线 AABB ± r          （笔触安全足迹上界）
    square  = 取较大轴 emax 的正方形  （等大 patch，供批处理）
    padded  = square 外扩 aabb_pad    （吸收 sigmoid 软过渡 / 数值误差）
    integer = ceil 整数像素框          （左上 ix0/iy0、边长 aabb_px，贴回原点）

注：``ix0/iy0`` 可越界（负 / 超 canvas），paste 时由调用方裁剪到画布内。
"""

import torch
from torch import Tensor


class AABBGeom:
    """曲线 AABB + r 的渲染框派生链（各级缓存）。

    Args:
        curve_aabb_min, curve_aabb_max: 曲线 AABB 下/上界，各 ``(B, 2)``（来自
            :class:`BezierGeom` / :class:`LineGeom` 的 ``.aabb``）。
        r_px: ``(B, 1)`` 或 ``(B,)`` 像素半径。
        aabb_pad: 正方形化外扩比例（吸收软过渡 / 数值误差），默认 0.0。

    Attributes:
        r_px: ``(B,)`` 像素半径（squeeze 至 1D，与 xmin/ymin 同维）。
        pad: 外扩比例。
        cx, cy: ``(B,)`` tube 中心（= square / padded 中心，整链不变）。
        ex, ey: ``(B,)`` tube 半宽 / 半高（**tube** stage）。
        emax: ``(B,)`` square 半宽 = max(ex, ey)（**square** stage）。
        half: ``(B,)`` padded 半宽 = emax * (1 + pad)（**padded** stage）。
        aabb_px: ``(B,)`` long 整数正方形边长 = ceil(2 * half)（**integer** stage）。
        ix0, iy0: ``(B,)`` long padded 左上 x/y（**integer** stage）。
    """

    def __init__(
        self,
        curve_aabb_min: Tensor,
        curve_aabb_max: Tensor,
        r_px: Tensor,
        aabb_pad: float = 0.0,
    ) -> None:
        xmin, xmax = curve_aabb_min[:, 0], curve_aabb_max[:, 0]   # (B,)
        ymin, ymax = curve_aabb_min[:, 1], curve_aabb_max[:, 1]   # (B,)
        rpx = r_px.squeeze(-1)                                    # (B,) 与 xmin/ymin 同维

        # ---- tube = 曲线 AABB ± r（笔触安全足迹上界）----
        tx0, tx1 = xmin - rpx, xmax + rpx
        ty0, ty1 = ymin - rpx, ymax + rpx
        cx = 0.5 * (tx0 + tx1)   # (B,) tube 中心
        cy = 0.5 * (ty0 + ty1)
        ex = 0.5 * (tx1 - tx0)   # (B,) tube 半宽
        ey = 0.5 * (ty1 - ty0)

        # ---- square = 较大轴 emax 的正方形（等大 patch，供批处理）----
        emax = torch.maximum(ex, ey)   # (B,)

        # ---- padded = square 外扩 aabb_pad（吸收 sigmoid 软过渡 / 数值误差）----
        half = emax * (1.0 + aabb_pad)   # (B,)

        # ---- integer = 整数像素框（patch 空间），对称 padded-anchored ----
        aabb_px = torch.clamp(torch.ceil(2.0 * half).long(), min=1)  # (B,) 正方形边长
        ix0 = torch.floor(cx - half).long()   # (B,) padded 左上 x
        iy0 = torch.floor(cy - half).long()   # (B,) padded 左上 y

        # 各级独立缓存
        self.r_px = rpx
        self.pad = aabb_pad
        self.cx, self.cy = cx, cy
        self.ex, self.ey = ex, ey
        self.emax = emax
        self.half = half
        self.aabb_px = aabb_px
        self.ix0, self.iy0 = ix0, iy0
