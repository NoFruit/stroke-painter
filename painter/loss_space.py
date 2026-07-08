"""loss_space.py - 项目侧组合损失空间（loss 计算环境，单一 cache owner）。

把"用哪几个 loss、权重多少、怎么组合"这一**项目级实验决策**从 loss 库里抽出来，
集中在项目侧。loss 库（losscal）只保留**无状态原语**（L1 / OT / Grad / Area），
每个就是 ``(pred, target) -> (scalar, info)`` 的忠实单次计算；本空间是它们的
**组合者 + 唯一 cache 持有者 + visual 提供者**。

为什么在项目侧、不在库里
------------------------
- "用哪几个 + 权重"是试错决策（SNP 用 L1+OT 即够，本项目想加 Grad/Area），应随手
  可改，不该冻死进库。
- 与笔刷侧对称：diffbrush 只给笔刷原语，"怎么渲染/合成"在项目 main。
- cache 天然属于组合者：单一 owner，不再每类各存一份。

组合语义（新设计：RGB + alpha 分离，非预乘 blend）
--------------------------------------------------
``forward(pred_canvas, target, brush_alpha=None)``：
    - pred_canvas : (B,3,H,W) RGB 当前画布状态（带梯度，mainloop 非预乘 blend 后的 canvas）
    - target      : (B,3,H,W) RGB 目标图（首次 forward 或 set_target 缓存，跨步复用）
    - brush_alpha : (B,1,H,W) 笔刷 alpha 覆盖图（AreaLoss 用；None 则跳过 area 项）
    返回组合标量 loss（可 backward 到 pred_canvas / brush_alpha）。

数据流对齐：
    笔刷 forward -> (color(B,3,H,W), alpha(B,1,H,w))
    mainloop   -> canvas = color*alpha + canvas*(1-alpha)  （非预乘 over 合成）
    loss_space -> L1/Grad/OT 吃 (canvas_rgb, target_rgb)；Area 吃 brush_alpha
    target 是纯 RGB（无 alpha 维），无需预乘转换。

cache 装什么
------------
反向传播不需要 cache（autograd 图自持中间量）；cache 唯一用途是**可视化**：
    - "total"     : 组合标量
    - "breakdown" : 各项标量 {l1, ot, grad, area}
    - 各原语的 visual 张量（OT 残差 / L1 差 / 梯度差 / alpha），由 viewer 渲染

target 跨步缓存
---------------
target 不变，loss_space 持有 ``self._target``；首次 forward 或 set_target 设置。
各原语本身无 target 缓存--由本空间每次传入。

范式：**不做防御性编程**。未先 set_target / 传 target 就 forward -> 自然报错（None
上的操作崩），不额外 guard。原语用错（如没传 info 给 visual）同理自然崩。

device 语义：纯透传。无自建张量、无 device 属性--pred_canvas/target/brush_alpha
在哪个 device，各原语就在哪个 device 算（losscal 原语均输入跟随）。target 缓存
仅 detach 不 .to，保持原 device。
"""

from typing import Dict, Optional

import torch
from torch import Tensor

import losscal


class LossSpace:
    """组合损失空间：统筹各 loss 原语，持有单一 cache，提供 visual。

    项与权重（硬编码，BofP L_app 子集 + SNP 经验）：
        l1   : PixelL1Loss     -- 像素差，主导/兜底（RGB）
        ot   : PointCloudOTLoss-- 质量分布匹配，远距离防梯度消失（RGB）
        grad : GradientLoss    -- 梯度对齐，笔触沿边缘走（RGB）
        area : AreaLoss        -- 笔触面积正则，防退化（需 brush_alpha）
    """

    def __init__(self):
        # ---- loss 原语实例（无状态，构造一次复用）----
        self.l1 = losscal.PixelL1Loss()
        self.ot = losscal.PointCloudOTLoss()
        self.grad = losscal.GradientLoss()
        self.area = losscal.AreaLoss()

        # ---- 权重（对齐 BofP 4.1 原文 L_app 几何重建权重）----
        # 原文：λ_pixel=1.0, λ_OT=0.2, λ_grad=0.1, λ_area=0.02
        # （λ_seg=0.1 / λ_perc=0.1 原文亦有，但需分割图/VGG，本项目无此原语，跳过）
        self.w_l1 = 1.0
        self.w_ot = 0.2
        self.w_grad = 0.1
        self.w_area = 0.02

        # ---- target 跨步缓存（target 不变）----
        self._target: Optional[Tensor] = None     # (B,3,H,W) RGB detach

        # ---- 单一 cache（本步可视化快照）----
        self._cache: Dict = {}

    # ------------------------------------------------------------------ #
    # target 跨步缓存
    # ------------------------------------------------------------------ #
    def set_target(self, target: Tensor) -> Tensor:
        """缓存 target（跨步不变，算一次复用）。

        target 为 RGB (B,3,H,W)，直接 detach 缓存--新设计 target 无 alpha 维，
        无需预乘转换（pred canvas 也是 RGB，非预乘 blend 产物，同表示直接比）。
        """
        self._target = target.detach().float()
        return self._target

    @property
    def target(self) -> Optional[Tensor]:
        return self._target

    # ------------------------------------------------------------------ #
    # 组合 forward（常用 loss，可 backward）+ 暂存 cache
    # ------------------------------------------------------------------ #
    def forward(
        self,
        pred_canvas: Tensor,
        target: Optional[Tensor] = None,
        brush_alpha: Optional[Tensor] = None,
    ) -> Tensor:
        """组合各 loss 原语，返回加权标量。

        Args:
            pred_canvas: (B,3,H,W) RGB 当前画布状态，带梯度（mainloop 非预乘 blend 后）。
            target: (B,3,H,W) RGB 目标图；None 则用已缓存 target（须先 set_target）。
            brush_alpha: (B,1,H,W) 笔刷 alpha 覆盖图（AreaLoss 用）；None 跳过 area。

        Returns:
            组合标量 loss，梯度可通到 ``pred_canvas``（及 ``brush_alpha``）。
            同时暂存本步 breakdown + 各 visual 到 ``self._cache``。
        """
        if target is not None:
            self.set_target(target)
        tgt = self._target
        if tgt is None:
            raise RuntimeError("未提供 target，且无缓存 target；请先 set_target。")

        breakdown = {}
        visuals: Dict[str, Tensor] = {}

        # ---- L1 (RGB) ----
        l_l1, info_l1 = self.l1.forward(pred_canvas, tgt)
        breakdown["l1"] = float(l_l1.detach())
        visuals.update(self._pref(self.l1.visual(info_l1), "l1"))

        # ---- OT (RGB) ----
        l_ot, info_ot = self.ot.forward(pred_canvas, tgt)
        breakdown["ot"] = float(l_ot.detach())
        visuals.update(self._pref(self.ot.visual(info_ot), "ot"))

        # ---- Grad (RGB) ----
        l_grad, info_grad = self.grad.forward(pred_canvas, tgt)
        breakdown["grad"] = float(l_grad.detach())
        visuals.update(self._pref(self.grad.visual(info_grad), "grad"))

        # ---- Area (alpha；需 brush_alpha；None 跳过）----
        if brush_alpha is not None:
            l_area, info_area = self.area.forward(brush_alpha, tgt)
            breakdown["area"] = float(l_area.detach())
            visuals.update(self._pref(self.area.visual(info_area), "area"))
        else:
            breakdown["area"] = None

        # ---- 加权组合 ----
        total = (
            self.w_l1 * l_l1
            + self.w_ot * l_ot
            + self.w_grad * l_grad
        )
        if brush_alpha is not None:
            total = total + self.w_area * l_area

        # ---- 暂存单一 cache ----
        self._cache = {
            "total": float(total.detach()),
            "breakdown": breakdown,
            "visuals": visuals,
        }
        return total

    @staticmethod
    def _pref(viz: Dict[str, Tensor], prefix: str) -> Dict[str, Tensor]:
        """给原语 visual 的键加前缀（避免跨原语同名冲突，如 l1/ot/grad 各有 diff）。"""
        return {f"{prefix}_{k}": v for k, v in viz.items()}

    # ------------------------------------------------------------------ #
    # 可视（读 cache，零额外计算）
    # ------------------------------------------------------------------ #
    def visual(self) -> Dict:
        """返回本步可视化结构（读 forward 的 cache，不重算任何原语）。

        Returns:
            dict：
                - "total"     : 组合标量
                - "breakdown" : 各项标量 {l1, ot, grad, area}
                - "visuals"   : 各原语 visual 张量（前缀化键，四维标准，detach）
        """
        if not self._cache:
            raise RuntimeError("无缓存：请先 forward(pred, target)。")
        return self._cache

    def cache_breakdown_str(self) -> str:
        """便捷：本步各项 breakdown 的可打印串（未 forward 则提示）。"""
        if not self._cache:
            return "<no cache; forward first>"
        bd = self._cache["breakdown"]
        return "  ".join(
            f"{k}={v:.4f}" if v is not None else f"{k}=skip"
            for k, v in bd.items()
        )


# 模块级全局单例：项目侧唯一 loss 计算环境，main 启动后随时可用。
loss_space = LossSpace()


__all__ = ["LossSpace", "loss_space"]
