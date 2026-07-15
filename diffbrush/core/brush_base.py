"""BrushBase - 笔刷抽象基类契约。

所有笔刷实现必须继承本类并实现 :meth:`forward`，遵循
"归一化参数 -> (color, alpha) 双输出" 的语义，且不得对输入参数施加任何约束。

输出约定（解耦，非预乘）：
    - ``color``  : ``(B, 3, H, W)`` 笔刷纯色 c 广播（**不乘覆盖度**）
    - ``alpha``  : ``(B, 1, H, W)`` 覆盖度 = α·M（笔刷 footprint 的 soft 掩码）

调用方（mainloop）按需组合：``canvas = color * alpha + canvas * (1 - alpha)``
（非预乘 alpha blend）。``color * alpha`` 即原预乘 RGB，可随时派生。
"""

from abc import ABC, abstractmethod
from typing import NamedTuple, Tuple

from torch import Tensor


class ParamLayout(NamedTuple):
    """笔刷参数的语义通道布局（供外部查询：Painter init/reparam/params_to_full 用）。

    所有笔刷参数共享同一语义结构 {几何控制点..., c, r, α}，差异仅在控制点数量
    （Line=2: P0,P2；Bezier=3: P0,P1,P2）与由此产生的通道偏移。本布局把"哪段通道
    是什么"暴露给外部，消除 main/Painter 里硬编码笔刷类型索引的必要。
    """
    param_dim: int                              # 总维数（Line=9, Bezier=11）
    point_slices: tuple                         # 几何控制点切片 ((s,e),...)，每个 2-dim
    color_slice: tuple                          # 颜色 RGB 切片 (s, e)，3-dim
    radius_idx: int                             # 半径标量索引
    alpha_idx: int                              # alpha 标量索引


class BrushBase(ABC):
    @classmethod
    @abstractmethod
    def param_layout(cls) -> ParamLayout:
        """笔刷参数的语义通道布局。

        供外部（Painter）查询"哪段通道是什么语义"，无需硬编码索引。
        生命周期内笔刷类型固定 -> 布局不变，可缓存。
        """
        ...

    @staticmethod
    @abstractmethod
    def unpack_params(params: Tensor):
        """解析参数为命名分量（子类定义返回结构，供 forward 内部用）。

        返回值结构与笔刷类型相关（Line 返回 5 元组，Bezier 返回 6 元组），
        故不固定签名；ABC 仅强制"笔刷必须能解析自身参数"这一范式。
        """
        ...

    @abstractmethod
    def forward(self, params: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """归一化笔刷参数 -> (color, alpha)。

        Args:
            params: 归一化后的笔刷参数张量，形状 ``(B, param_dim)``，含义由子类定义。
            *args, **kwargs: 实现类自定义的额外输入（必须包含 patch 信息：
                patch 大小或画布引用，具体形式由实现者选择）。

        Returns:
            (color, alpha)：color 为 ``(B, 3, H, W)`` 笔刷纯色（非预乘），
            alpha 为 ``(B, 1, H, W)`` 覆盖度（α·M）；梯度均可通到 ``params``。
        """
        ...

    @abstractmethod
    def forward_fast(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        ...

    def __call__(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """调用即 forward（轻量转发，不引入 nn.Module 开销）。"""
        return self.forward(*args, **kwargs)
