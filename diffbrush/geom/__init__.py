"""diffbrush.geom - 纯几何生产者包。

把 :mod:`diffbrush.utils` 的纯数学函数包成「构造时算一次、缓存到 self」的
几何对象，供绘制管线（``render_coverage`` 等）上游复用。本包只产参数、不碰绘制；
math 留在 utils 不动。

五个类：

- :class:`BezierGeom` / :class:`LineGeom`: 曲线 1D 解析式（r 无关），含曲线 AABB
  与弧长采样 ``.sample``。
- :class:`StampGeom`: 绘制前夕的几何无关数据 = 采样集 + 半径（不知曲线类型）。
- :class:`AABBGeom`: 曲线 AABB + r -> 渲染框派生链（tube/square/padded/integer）。
- :class:`CapsGeom`: 方头端点切线 u0/u2（r 无关，仅 square）。

详见 ``geom/README.md`` 开发文档。
"""

from .aabb_geom import AABBGeom
from .bezier_geom import BezierGeom
from .caps_geom import CapsGeom
from .line_geom import LineGeom
from .stamp_geom import StampGeom

__all__ = [
    "BezierGeom",
    "LineGeom",
    "StampGeom",
    "AABBGeom",
    "CapsGeom",
]
