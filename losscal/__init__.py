"""losscal — 以 geomloss 为计算核心的损失计算原语模组。

四原语（统一契约：无状态 ``forward→(scalar, info)`` + ``visual(info)→dict``）：

    - :class:`PixelL1Loss`       像素 L1 距离（RGB 逐像素差）
    - :class:`PointCloudOTLoss`  点云 debiased Sinkhorn OT 散度（geomloss 点云路径）
    - :class:`GradientLoss`      梯度对齐（Sobel 幅值差 + 方向差）
    - :class:`AreaLoss`          笔触面积正则（alpha 覆盖度）

设计要点
--------
- **原语无状态**：``self`` 不持 cache、不缓存 target。``forward`` 忠实单次计算，
  返回 ``(scalar, info)``；``visual`` 是纯函数，吃 info 组装 viz（不读 self）。
  组合 / 加权 / target 跨步复用由上层持有，单一 cache owner。
- **geomloss 延迟导入**：只在 :class:`PointCloudOTLoss` 实例化时触发，
  L1 / Grad / Area 不被 geomloss 拖累——``import losscal`` 本身不依赖 geomloss。

公共 API：

    from losscal import (
        LossBase,
        PixelL1Loss,
        PointCloudOTLoss,
        GradientLoss,
        AreaLoss,
    )
"""

from .loss_base import LossBase
from .pixel_l1_loss import PixelL1Loss
from .gradient_loss import GradientLoss
from .area_loss import AreaLoss
from .point_cloud_ot_loss import PointCloudOTLoss

__all__ = [
    "LossBase",
    "PixelL1Loss",
    "PointCloudOTLoss",
    "GradientLoss",
    "AreaLoss",
]

__version__ = "0.1.0"
