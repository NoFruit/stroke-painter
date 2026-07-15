"""painter.py - Painter 类：coarse-to-fine 笔画优化绘制器。

仿旧 main.py 的成熟结构（已弃用，保留作参考），封装为类。main 的三层循环
（level -> stroke-batch -> epoch）+ 重参数化 + init_raw + commit 逻辑将逐步迁入。

本阶段：构造上下文 + 优化空间映射 + 渲染 + init_raw + 三层循环 全部就位。

新设计数据流（与旧 main 4 通道预乘 RGBA 不同）：
    笔刷 forward -> (color(B,3,H,W), alpha(B,1,H,W))
    mainloop   -> canvas = color*alpha + canvas*(1-alpha)  （非预乘 over 合成）
    loss       -> L1/Grad/OT(canvas_rgb, target_rgb)；Area(alpha)
    canvas     : RGB (1,3,H,W)（无 alpha 维）

device 语义（全包统一）
-----------------------
    Painter 是 device 的唯一外部入口--构造时必传 device（无 auto-detect）。
    包内各组件的 device 归属：
      ImageInput            : 由 Painter 传入 device，用于从文件建张量（PIL->tensor、
                               zeros canvas）。是包内唯一"凭空建张量"的组件。
      CoarseToFine / ErrorMap: 构造时收 target，device 跟 target 走（输入跟随）。
      LossSpace / brush      : 纯透传，device 跟输入张量走（无自建张量）。
    包内零 device 解析逻辑（无 auto-detect、无 device.py）。
"""

import torch

# ---- painter 包内协助工具（相对导入，扁平包范式）----
from .image_input import ImageInput
from .coarse_to_fine import CoarseToFine
from .error_map import ErrorMap
from .loss_space import LossSpace

# ---- 外部包（扁平并列：diffbrush / losscal 经 loss_space 间接用）----
import diffbrush


def _resolve_brush_params(brush: diffbrush.BrushBase) -> diffbrush.ParamLayout:
    """询问笔刷的参数语义布局。Painter 生命周期内笔刷类型固定 -> 布局不变，缓存即可。

    消除 main.py 里硬编码笔刷类型索引（"Line 9-dim / Bezier 11-dim"）的必要：
    init_raw / reparam / params_to_full 均通过本布局查询通道位置，与笔刷类型解耦。
    """
    return brush.param_layout()


class Painter:
    """coarse-to-fine 笔画优化绘制器。

    仿旧 main.py 结构：coarse-to-fine 三层循环（level -> stroke-batch -> epoch）
    + ErrorMap 矩引导 init + 重参数化约束 + 批化 commit。逐步从 main 迁入。

    构造上下文（__init__ 就位，仿 main setup 阶段）：
        device       : 构造必传（torch.device），Painter 不做 auto-detect。
                       传给 ImageInput 建张量；c2f/em 跟 target.device；loss_space/brush 跟输入。
        img_input    : 图像输入实例（已 load）
        target_rgb   : (1,3,H,W) 目标图 RGB
        target_alpha : (1,1,H,W) 目标图 alpha（无 alpha 源补全 1）
        canvas       : (1,3,H,W) 当前画布 RGB（空白全 0，累积载体）
        brush        : 可微笔刷（diffbrush，forward -> (color, alpha) 元组）
        param_layout : 笔刷参数语义布局（询问 brush.param_layout() 缓存；init/reparam/params_to_full 用）
        c2f          : CoarseToFine 金字塔切片器（RGB 3 通道，跟随 target device）
        em           : ErrorMap 误差矩特征（RGB 统一并行）
        loss_space   : LossSpace 组合损失空间（L1/OT/Grad/Area，RGB+alpha 契约）

    已迁入（param_layout 驱动，笔刷类型无关）：
        - reparam / _ste_clamp（raw -> 合法域：c/α STE clamp，r exp）
        - params_to_full（slice 归一 -> 原图归一，统一 Line/Bezier）
        - _forward_strokes / _commit_strokes（新 (color,alpha) 契约 + 非预乘 blend）
        - init_raw（ErrorMap 矩特征 -> 笔刷参数 init；几何按控制点数量分支）
        - run（coarse-to-fine 三层优化循环）

    未迁入：
        - BrushStyleApplicator（neube 未迁入，风格化备用，暂缺）
    """

    def __init__(self, device: torch.device):
        # ---- device（外部传入，Painter 不解析；传给 ImageInput 建张量）----
        self.device = device

        # ---- 图像输入：load target_rgb + target_alpha + canvas_rgb ----
        self.img_input = ImageInput(self.device).load()
        self.target_rgb = self.img_input.target_rgb         # (1,3,H,W) RGB 或 None
        self.target_alpha = self.img_input.target_alpha     # (1,1,H,W)（无 alpha 源补全 1）
        self.canvas = self.img_input.canvas_rgb             # (1,3,H,W) RGB 空白全 0（累积载体）

        # ---- 笔刷（diffbrush，forward -> (color, alpha) 元组）----
        self.brush = diffbrush.BezierSquareBrush()

        # ---- 笔刷参数布局（询问笔刷，缓存；生命周期内笔刷类型固定 -> 不变）----
        self.param_layout = _resolve_brush_params(self.brush)

        # ---- coarse-to-fine 金字塔切片器（RGB 3 通道，跟随 target device）----
        self.c2f = CoarseToFine(self.target_rgb)

        # ---- ErrorMap 误差矩特征（RGB 统一并行，init 缓存 target 全图）----
        self.em = ErrorMap(self.target_rgb)

        # ---- 组合损失空间（L1/OT/Grad/Area，RGB+alpha 契约，单一 cache owner）----
        self.loss_space = LossSpace()

        # ---- 优化配置（硬编码，全项目范式）----
        self.n_strokes = 5              # 每级笔画批数
        self.epochs_per_stroke = 64    # 每批优化步数
        self.opt_lr = 0.003             # RMSprop 学习率（对齐 BofP 4.1 几何重建）
        self.commit_chunk = 8           # commit 批化 chunk 大小（显存/并行权衡）

        # ---- init_raw 超参（ErrorMap 矩引导 -> 笔刷参数 init）----
        self.init_seg_div = 3.0         # 主轴三段等分：u∈[-A,A] -> 两端段/中段
        self.init_r_lo = 0.01           # r 地板：B->0 时防退化
        self.init_color_jitter = 0.05   # 颜色抖动半幅：c = top_color ± U(jitter)
        self.init_alpha = 0.8           # α 固定初值（STE clamp [0,1]，raw 直接赋）

    # ================================================================== #
    # 优化空间映射（raw <-> 合法域）
    # ================================================================== #

    @staticmethod
    def _ste_clamp(x: torch.Tensor, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
        """STE clamp：forward = clamp(x, lo, hi)，backward = identity（雅可比 = 1）。

        消除 sigmoid 雅可比 c(1-c) 在 0/1 附近饱和导致颜色优化卡死的问题。
        """
        return x + (x.clamp(lo, hi) - x).detach()

    def reparam(self, raw: torch.Tensor) -> torch.Tensor:
        """raw (B,D) 无约束 -> params (B,D) 物理量落入合法域，全程可微。

        通道约束由 param_layout 驱动（笔刷类型无关）：
          几何点    : 恒等透传（自由漂移，越界中心不贡献覆盖）
          颜色 c    : STE clamp [0,1]
          半径 r    : exp（positive 双射，防 r<0 -> NaN）
          alpha α   : STE clamp [0,1]
        """
        layout = self.param_layout
        out = raw.clone()
        cs, ce = layout.color_slice
        out[:, cs:ce] = self._ste_clamp(raw[:, cs:ce])
        out[:, layout.radius_idx] = torch.exp(raw[:, layout.radius_idx])
        out[:, layout.alpha_idx] = self._ste_clamp(raw[:, layout.alpha_idx])
        return out

    # ================================================================== #
    # 坐标空间互转（slice 归一 <-> 原图归一）
    # ================================================================== #

    def params_to_full(self, params: torch.Tensor, level) -> torch.Tensor:
        """切片归一 params -> 原图归一 params（用 c2f transform + param_layout）。

        几何点 : 逐点 slice 用 CoarseToFine.transform_point 做 affine 变换
        半径 r : 用 CoarseToFine.transform_radius 做 r_scale 变换
        颜色/α : 不变（与坐标空间无关）

        统一 Line / Bezier：控制点数量由 param_layout.point_slices 决定。
        """
        layout = self.param_layout
        aff = level.transform.point_affine
        rs = level.transform.r_scale
        p = params.clone()
        for (s, e) in layout.point_slices:
            p[:, s:e] = CoarseToFine.transform_point(params[:, s:e], aff)
        p[:, layout.radius_idx] = CoarseToFine.transform_radius(
            params[:, layout.radius_idx], rs)
        return p

    # ================================================================== #
    # 渲染（新 (color, alpha) 契约 + 非预乘 over blend）
    # ================================================================== #

    def _forward_strokes(self, params: torch.Tensor, canvas_batch: torch.Tensor,
                         patch_size) -> tuple:
        """一批笔画在 canvas 切片上绘制（带梯度，非预乘 over blend）。

        params       : (b, D) 归一化笔刷参数（带梯度）
        canvas_batch : (b, 3, ph, pw) RGB canvas 切片
        patch_size   : (ph, pw)
        返回:
            composited  : (b, 3, ph, pw) 合成后 RGB
            brush_alpha : (b, 1, ph, pw) 本批各笔 alpha 覆盖图
        """
        color, alpha = self.brush.forward(params, patch_size)
        composited = color * alpha + canvas_batch * (1.0 - alpha)
        return composited, alpha

    def _commit_strokes(self, params_full: torch.Tensor,
                        canvas_full: torch.Tensor) -> torch.Tensor:
        """正式光栅化：把一批笔画（原图空间）画回全图画布（无梯度，chunk 化串行 over）。

        params_full : (B, D) 原图归一参数（detach）
        canvas_full : (1, 3, H, W) 全图画布 RGB（累积载体）
        返回:
            canvas_full : (1, 3, H, W) 累积后的画布

        chunk 化 forward（渲染重活批化）+ 组内逐笔串行 over 合成保序（非预乘 over
        不可交换：重叠区按笔顺序叠加）。
        """
        B = params_full.shape[0]
        H, W = canvas_full.shape[-2], canvas_full.shape[-1]
        with torch.no_grad():
            for i in range(0, B, self.commit_chunk):
                j = min(i + self.commit_chunk, B)
                color, alpha = self.brush.forward_fast(params_full[i:j], (H, W))
                for k in range(j - i):
                    c_k = color[k:k + 1]
                    a_k = alpha[k:k + 1]
                    canvas_full = c_k * a_k + canvas_full * (1.0 - a_k)
        return canvas_full

    # ================================================================== #
    # ErrorMap 引导 init
    # ================================================================== #

    def init_raw(self, b: int) -> torch.Tensor:
        """ErrorMap 矩特征 -> 初始 raw（无约束），使 reparam(raw) 还原到矩引导初值。

        每片一个 top 块（em 选出的“最该补的块”）-> 一笔。把 top 块的质心 / 走向 /
        主次轴半长 / 块内 target 均色 映射成笔刷参数装 raw。

        几何生成由 param_layout.point_slices 数量驱动（笔刷类型无关）：
          首点 : u ∈ [-A, -A/3]  (OBB 一端段)
          末点 : u ∈ [ A/3,  A]  (OBB 另一端段)
          中间点: u ∈ [-A/3, A/3] (中间段；Bezier P1 沿主轴插值)
        所有点 v ∈ [-B, B] 垂直全自由。Line(2点)=首末；Bezier(3点)=首中末。

        颜色/α 用 STE clamp（非双射）：raw 直接赋域内初值。
        r 用 exp（双射）：raw = log(r_init)。
        """
        layout = self.param_layout
        em = self.em
        dev = self.device

        # ---- 矩特征 -> 归一 OBB（patch 像素 [0,S-1] -> 归一 [0,1]，÷S）----
        S = em.labels.shape[-1]
        cen = em.top_centroid / S                    # (B,2)
        A = em.top_axis_major / S                    # (B,) 主轴半长
        B = em.top_axis_minor / S                    # (B,) 次轴半长
        th = em.top_orientation                      # (B,) 走向
        d = torch.stack([th.cos(), th.sin()], dim=1) # (B,2) 主轴方向
        p = torch.stack([-th.sin(), th.cos()], dim=1)# (B,2) 垂直方向

        # ---- 逐控制点生成（首末两端段，中间点中段）----
        n_pts = len(layout.point_slices)
        A3 = A / self.init_seg_div
        pts_list = []
        for i in range(n_pts):
            if i == 0:
                u = -A + 2.0 * A3 * torch.rand(b, device=dev)     # U[-A, -A/3]
            elif i == n_pts - 1:
                u = A3 + 2.0 * A3 * torch.rand(b, device=dev)     # U[A/3, A]
            else:
                u = -A3 + 2.0 * A3 * torch.rand(b, device=dev)    # U[-A/3, A/3]
            v = B * (2.0 * torch.rand(b, device=dev) - 1.0)       # U[-B, B]
            P = cen + u[:, None] * d + v[:, None] * p             # (B,2)
            pts_list.append(P.clamp(0, 1))

        # ---- r: U[B/2, B]，笔宽 2r ∈ U[b,2b] 不溢出 OBB 次轴 ----
        B_lo = (B * 0.5).clamp(min=self.init_r_lo)
        B_hi = B.clamp(min=self.init_r_lo)
        r_init = B_lo + (B_hi - B_lo) * torch.rand(b, device=dev)

        # ---- 颜色: top_color ± jitter ----
        c = (em.top_color
             + self.init_color_jitter
             * (2.0 * torch.rand(b, 3, device=dev) - 1.0)).clamp(0.0, 1.0)

        # ---- α 固定 ----
        a_init = torch.full((b,), self.init_alpha, device=dev)

        # ---- 装 raw（param_layout 驱动，笔刷类型无关）----
        raw = torch.empty(b, layout.param_dim, device=dev)
        for i, (s, e) in enumerate(layout.point_slices):
            raw[:, s:e] = pts_list[i]
        cs, ce = layout.color_slice
        raw[:, cs:ce] = c                                   # STE: raw_c = c0（clamp(c0)=c0）
        raw[:, layout.radius_idx] = torch.log(r_init)       # exp 的逆
        raw[:, layout.alpha_idx] = a_init                   # STE: raw_α = α0
        raw.requires_grad_(True)
        return raw

    # ================================================================== #
    # coarse-to-fine 三层优化循环
    # ================================================================== #

    def run(self):
        """coarse-to-fine 三层优化循环（level -> stroke-batch -> epoch）。

        level 循环      ：金字塔逐级（粗->细），canvas_full 跨级累积。
        stroke-batch 循环：每级 n_strokes 批，每批 B=grid_n² 笔（一片一笔）。
        epoch 循环      ：epochs_per_stroke 步优化本批 raw（reparam -> render -> loss -> backward）。

        每批优化完：
          1) 固化到 canvas_batch（tile-res，保持优化器所见分辨率）。
          2) commit：params_to_full 翻译 -> _commit_strokes 光栅化到 canvas_full（full-res）。

        结束：落盘最终画布。无可视化（无优化版本）。
        """
        canvas_full = self.canvas.clone()                       # (1,3,H,W) RGB 累积载体

        for li, lvl in enumerate(self.c2f.pyramid):
            b = lvl.n_tiles
            patch_size = lvl.patch_hw
            target_batch = lvl.image                            # (B,3,ph,pw) RGB 监督量
            canvas_batch = self.c2f.slice(canvas_full)[li].image  # (B,3,ph,pw) RGB 当前级画布

            print(f"[level {li+1}/{len(self.c2f.pyramid)}] "
                  f"grid={lvl.grid_n} B={b} patch={patch_size}")

            for si in range(self.n_strokes):
                # ---- 算误差 -> 矩引导 init ----
                self.em.compute(canvas_batch, target_batch)
                raw = self.init_raw(b)
                optimizer = torch.optim.RMSprop([raw], lr=self.opt_lr)

                # ---- epoch 循环：优化 raw ----
                last_loss = None
                for ei in range(self.epochs_per_stroke):
                    optimizer.zero_grad()
                    params = self.reparam(raw)
                    composited, alpha = self._forward_strokes(
                        params, canvas_batch, patch_size)
                    loss = self.loss_space.forward(
                        composited, target_batch, alpha)
                    loss.backward()
                    optimizer.step()
                    last_loss = float(loss.detach())

                # ---- 固化本批：reparam 出合法 params -> 更新 canvas_batch ----
                params = self.reparam(raw).detach()
                with torch.no_grad():
                    composited, _ = self._forward_strokes(
                        params, canvas_batch, patch_size)
                    canvas_batch = composited

                # ---- commit：翻译到原图空间 -> 光栅化到正式画布 ----
                params_full = self.params_to_full(params, lvl)
                canvas_full = self._commit_strokes(params_full, canvas_full)

                print(f"  batch {si+1}/{self.n_strokes}  loss={last_loss:.6f}")

        # ---- 落盘 ----
        out_path = ImageInput.save_output(canvas_full)
        print(f"output: {out_path}")
        return canvas_full
