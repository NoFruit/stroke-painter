"""coarse_to_fine.py — coarse-to-fine 子画布金字塔（优化空间类）。

定位
----
BofP coarse-to-fine 是**优化空间**类型：把全图按 1×1→2×2→...→n×n 等分网格切
片，每一级每一片是本笔优化的 patch（笔刷归一空间）。粗级切片大→过 cap 则降采样
到 cap；细级切片小→直接原图精度刻画细节。

本类**只做"切图 + 降采样 + 坐标置换"**——coarse-to-fine 的空间几何精髓，不含
笔刷/loss/优化（那是后续阶段）。本文件 main 即此阶段唯一测试场。

数据二元论（参数维度无关）
-------------------------
每个切片 `Level` 只管两份数据，**不感知笔刷参数是几维、哪几通道是点 / 哪个是 r**：

- **切片** `image`  : (B,3,ph,pw) 像素（监督量 / canvas 底子），进 optimizer。
- **置换** `transform` : `TileTransform`——切片归一 [0,1]² ↔ 原图归一 [0,1]² 的坐标
  互转信息（`point_affine (B,2,3)` + `r_scale (B,)`）。纯数据，调用方自取自用。

坐标转换走两个原子静态方法（维度无关，外部自行取参数通道）：
- ``transform_point(point_xy, point_affine)`` — 单点 (x,y) 切片归一→原图归一
- ``transform_radius(r, r_scale)`` — 半径标量缩放

通用切片流程（对任意分辨率 target）
--------------------------------
1. **levels 自动生成**：输入 n_target（目标最细网格边长）+ factor（逐级倍率，默认 2）。
   规约到最大的 factor^k ≤ n_target 且整除 min(H,W)，levels=[1, factor, …, factor^k]。
   factor=2 即图像金字塔标准 factor-2（Gaussian/Laplacian/mipmap/SIFT/FPN 惯例）。
   网格边长数列 1×1→2×2→…→n×n，对应 BofP 描述。
2. 等分切图：n×n n²片（行主序，idx = row*n + col）。要求 n 整除 min(H,W)（由规约保证）。
3. 单切片降采样规则：切片原图精度为 region×region；若 **region > cap**，则该
   切片降采样到 **cap×cap**；若 **region ≤ cap**，则直接用原精度（region×region）。
   ——即"超过才降"，未超不放大、不在细级凭空超采样造细节。
   （32×32 + 1×1：region=32 ≤ cap=128 → 不降，patch 仍是 32；切片填不满 cap 属正常）
4. 每个切片记一个坐标置换 `TileTransform`：把笔刷在"切片归一空间 [0,1]²"的坐标映射
   回"原图归一空间 [0,1]²"，供后续把优化出的笔刷画回全分辨率画布。

坐标置换约定（TileTransform）
----------------------------
笔刷参数的 P0/P1/P2 等点坐标在 [0,1]² 内，语义为"本切片内的归一坐标"。要画回原图
归一空间，外部自行取参数的 x,y 通道，调用 ``CoarseToFine.transform_point`` 做单点转换；
取 r 通道调用 ``CoarseToFine.transform_radius`` 做半径缩放。颜色/透明度等非几何参数不变。
本阶段**只产出置换数据，不画**；后续阶段按各切片 transform 落回原图即可。

范式
----
- **不做防御性编程**：target None / 维度错 → 自然崩，不 guard。
- **不读图入上下文**：本类只持有张量；main 用 plt 弹窗展示切片金字塔（**不落盘**，
  plt 弹窗不会把图塞进 agent 上下文）。
- **四维范式**：target (1,3,H,W) RGB；切片 (1,3,ph,pw)；置换用 2×3 仿射
  （供 affine_grid/后续渲染用）。切片/降采样/仿射均只作用于空间 (H,W)，与通道数无关。
- **config 参数化**：n_target/factor/cap 为构造参数（带默认值=现状值），向后兼容。

运行（本阶段测试）：
    cd ot-brush-optimize-workspace
    python coarse_to_fine.py
"""

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F


@dataclass
class TileTransform:
    """坐标置换组（纯数据，参数维度无关）。

    切片归一 [0,1]² ↔ 原图归一 [0,1]² 的变换数据。不含任何笔刷参数通道语义——
    外部自行读取 point_affine / r_scale，按自己的参数布局应用。

    - point_affine : (B,2,3) 2D 仿射 [[a,b,tx],[c,d,ty]]
                    应用到参数中的 2D 控制点：x'=a·x+b·y+tx, y'=c·x+d·y+ty
    - r_scale      : (B,) 半径标量缩放（= a = 切片→全图像素尺度比）
                    乘到参数中的半径标量上保像素半径一致

    颜色 / 透明度等非几何参数不在此组（不变）。
    等分网格 a=d，r_scale 取 a 无歧义；各向异性切片的 r 语义待定（本重构不涉及）。
    """
    point_affine: torch.Tensor   # (B,2,3)
    r_scale: torch.Tensor        # (B,)


@dataclass
class Level:
    """一个金字塔级：n×n 网格下所有切片，**批张量化**（固定格式 (B,3,ph,pw)）。

    数据二元论——只管两份数据，参数维度无关（不感知笔刷参数布局）：
        image      : (B,3,ph,pw) RGB float [0,1]，B = grid_n²。target=监督量/canvas=
                     断点载体，皆 detach。**本层主数据，固定 (B,3,ph,pw)，进 optimizer。**
        transform  : TileTransform——坐标置换（point_affine (B,2,3) + r_scale (B,)）。
                     切片归一 [0,1]² ↔ 原图归一 [0,1]² 的互转信息，纯数据外提，调用方自取自用。
    派生 property（保留 → 调用方零改）：
        n_tiles    : image.shape[0] = grid_n²。
        patch_hw   : (image.shape[-2], image.shape[-1]) 同级一致（批张量化前提）。

    砍掉的旧字段：row / col / region_full / downsampled / affine_full
    （affine_full → transform.point_affine）。
    """
    grid_n: int
    image: torch.Tensor            # (B,3,ph,pw)
    transform: TileTransform       # 坐标置换：point_affine + r_scale

    @property
    def n_tiles(self) -> int:
        return self.image.shape[0]

    @property
    def patch_hw(self) -> tuple:
        return (self.image.shape[-2], self.image.shape[-1])


class CoarseToFine:
    """coarse-to-fine 子画布金字塔（优化空间类）。

    创建即对 target 做预处理：自动生成 levels → 逐级切片 + 按规则降采样 + 算好每片
    坐标置换（TileTransform）。target 与 canvas 共用同一套切片逻辑（``_slice``），
    保证几何一致；canvas 切片走对外入口 ``slice(img)``（要求与 target 同尺寸）。
    之后可逐级迭代，每片作为一个独立优化 patch。

    config（构造参数，带默认值=现状值，直接可见、向后兼容）:
        n_target : 目标最细网格边长（n×n 的 n）。默认 16。
        factor   : 网格逐级倍率。默认 2 = 图像金字塔标准 factor-2。
        cap      : 单切片像素边长上限；region > cap 才降采样到 cap。默认 128。
    target  : (1,3,H,W) 全精度目标图 RGB（float [0,1]），计算空间三通道。
    device  : 跟 target 走（输入跟随）。affine/切片均在 target.device 上建。
    """

    def __init__(self, target: torch.Tensor):
        # ---- config（提为构造参数，默认=现状值，直接可见）----
        self.n_target = 16  # 目标最细网格边长（n×n 的 n）；将规约到 factor^k 且整除 min(H,W)
        self.factor = 2      # 网格逐级倍率。2 = 图像金字塔标准 factor-2（每级切片尺寸 ÷2）：
                                  # Gaussian/Laplacian pyramid、mipmap、SIFT、FPN 数十年惯例；且
                                  # BofP 的 128 StyleGAN patch 在 8×8 级 region 恰为 128，天然对齐。
                                  # 线性 1,2,3,... 级数爆炸、相邻级冗余、落不到 128，不推荐。
        self.cap = 128            # 单切片像素边长上限；region > cap 才降采样到 cap

        # ---- 暂存 target（全精度，detach；监督量不参与梯度）----
        if target is None:
            raise RuntimeError("CoarseToFine 强需求 target，got None。")
        t = target.detach().float()
        if t.dim() != 4 or t.shape[1] != 3:
            raise RuntimeError(
                f"target 需为 (1,3,H,W) RGB，got {tuple(t.shape)}。"
            )
        self.target = t
        self.H, self.W = t.shape[-2], t.shape[-1]   # 原图全精度尺寸

        # ---- levels 自动生成（由 min(H,W) 驱动，不写死尺寸）----
        self.levels = self._compute_levels()

        # ---- 预处理：逐级切片 + 降采样 + 算 affine ----
        # target 与 canvas 走同一套切片逻辑（_slice），保证几何一致（同 H,W → 同切法）。
        self.pyramid: List[Level] = self._slice(self.target)

    # ------------------------------------------------------------------ #
    # levels 自动生成
    # ------------------------------------------------------------------ #
    def _compute_levels(self) -> List[int]:
        """由 n_target / factor / (H,W) 自动生成网格边长数列（粗→细）。

        规约：找最大 k 使 factor^k ≤ n_target 且**同时整除 H 与 W**；
        levels = [factor^0, factor^1, …, factor^k]。
        要求整除 H 与 W（非仅 min）——保证同级所有切片尺寸完全一致，可堆成批张量
        (B,3,ph,pw)。对标准卷积尺寸（128/256/512/1024，均 2 幂）天然整除。
        """
        H, W = self.H, self.W
        f = self.factor
        if f < 2:
            raise RuntimeError(f"factor 须 ≥2，got {f}。")
        levels = [1]
        g = 1
        while g * f <= self.n_target and H % (g * f) == 0 and W % (g * f) == 0:
            g *= f
            levels.append(g)
        return levels

    # ------------------------------------------------------------------ #
    # 切片（target 与 canvas 共用同一套切法）
    # ------------------------------------------------------------------ #
    def _slice(self, img: torch.Tensor) -> List[Level]:
        """对任意 (1,3,H,W) 图像做逐级切片 → 降采样 → 记坐标置换，返回各级 Level。

        几何（patch_hw / transform）仅由 (H,W,config) 决定，与 img 内容无关；故对
        target 与同尺寸 canvas 切出的几何完全一致——这是"同一套切法"的保证。img
        仅决定每片的 image 像素。

        输出每级为**批张量** Level：image (B,3,ph,pw)、transform (TileTransform)
        （B=grid_n²）。同级切片尺寸一致（_compute_levels 保证 g 整除 H、W），故可堆批。
        affine 建在 img.device → transform 全程在 target device（修现状 CPU 瑕疵，
        params_to_full 的 .to 成无开销对齐）。
        """
        H, W = self.H, self.W
        levels: List[Level] = []
        for g in self.levels:
            if g <= 0:
                raise RuntimeError(f"level grid_n 须正整数，got {g}。")
            patches = []        # 每片 (1,3,ph,pw)，堆批前收集
            affines = []        # 每片 (1,2,3)
            # 同级等分（g 整除 H、W → 每片精确 step，无余数）
            step_r = H // g
            step_c = W // g
            region_norm = (step_r + step_c) / 2.0   # 归一用 step 均值（H=W 时即 step）
            for row in range(g):
                for col in range(g):
                    y0 = row * step_r
                    x0 = col * step_c
                    y1 = (row + 1) * step_r
                    x1 = (col + 1) * step_c
                    region_full = max(y1 - y0, x1 - x0)   # H=W 时即 step

                    # 切图（detach；target=监督量，canvas=断点载体，皆不参与梯度）
                    crop = img[:, :, y0:y1, x0:x1]            # (1,3,h,w)

                    # 降采样规则：region > cap 才降到 cap；否则原精度
                    if region_full > self.cap:
                        patch = F.adaptive_avg_pool2d(crop, (self.cap, self.cap))
                    else:
                        patch = crop

                    # point_affine：切片归一 [0,1]² → 原图归一 [0,1]²
                    #   x_full = x0/W + x_tile * (region_norm/W)
                    #   y_full = y0/H + y_tile * (region_norm/H)
                    #   → a=region_norm/W, d=region_norm/H, tx=x0/W, ty=y0/H, b=c=0
                    a = region_norm / W
                    d = region_norm / H
                    tx = x0 / W
                    ty = y0 / H
                    affine = torch.tensor(
                        [[a, 0.0, tx],
                         [0.0, d, ty]],
                        dtype=torch.float32,
                        device=img.device,
                    )[None]   # (1,2,3)  建在 img.device → transform 全程在 target device

                    patches.append(patch.detach())
                    affines.append(affine)

            point_affine = torch.cat(affines, dim=0)          # (B,2,3)
            r_scale = point_affine[:, 0, 0]                   # (B,) = a = 切片→全图像素尺度比
            levels.append(Level(
                grid_n=g,
                image=torch.cat(patches, dim=0),              # (B,3,ph,pw)
                transform=TileTransform(point_affine, r_scale),
            ))
        return levels

    def slice(self, img: torch.Tensor) -> List[Level]:
        """对外切片入口：用与 target **完全相同的切法**切任意 (1,3,H,W) 图像。

        canvas 切片用此：要求 img 与 target 同 (H,W)（否则几何不一致，断点重续/画回全图
        会错位），返回各级 Level（几何与 self.pyramid 一致，仅 image 像素不同）。
        """
        if img is None:
            raise RuntimeError("slice 强需求 img，got None。")
        t = img.detach().float()
        if t.dim() != 4 or t.shape[1] != 3:
            raise RuntimeError(f"img 需为 (1,3,H,W) RGB，got {tuple(t.shape)}。")
        if t.shape[-2:] != (self.H, self.W):
            raise RuntimeError(
                f"img 尺寸须与 target 一致 {(self.H, self.W)}，got {tuple(t.shape[-2:])}。"
            )
        return self._slice(t)

    # ------------------------------------------------------------------ #
    # 坐标置换原子工具（静态，维度无关）
    # ------------------------------------------------------------------ #
    @staticmethod
    def transform_point(point_xy: torch.Tensor, point_affine: torch.Tensor) -> torch.Tensor:
        """单点坐标置换：切片归一 [0,1]² → 原图归一 [0,1]²。

        维度无关——不感知笔刷参数布局。外部自行取出参数的 x,y 通道传入。

        Args:
            point_xy:     (*, 2)      x,y 在切片归一空间 [0,1]²
            point_affine: (*, 2, 3)  仿射 [[a,b,tx],[c,d,ty]]

        Returns:
            (*, 2)  x,y 在原图归一空间 [0,1]²
        """
        a = point_affine[..., 0, 0]
        b = point_affine[..., 0, 1]
        tx = point_affine[..., 0, 2]
        c = point_affine[..., 1, 0]
        d = point_affine[..., 1, 1]
        ty = point_affine[..., 1, 2]
        x, y = point_xy[..., 0], point_xy[..., 1]
        x_full = a * x + b * y + tx
        y_full = c * x + d * y + ty
        return torch.stack([x_full, y_full], dim=-1)

    @staticmethod
    def transform_radius(r: torch.Tensor, r_scale: torch.Tensor) -> torch.Tensor:
        """半径缩放：切片归一 → 原图归一。

        维度无关——外部自行取出参数的 r 通道传入。

        Args:
            r:       (*,)  半径标量（切片归一空间）
            r_scale: (*,)  半径缩放因子（= a = 切片→全图像素尺度比）

        Returns:
            (*,)  半径在原图归一空间
        """
        return r * r_scale

    # ------------------------------------------------------------------ #
    # 参数翻译：切片归一空间 → 原图归一空间（已移除）
    # ------------------------------------------------------------------ #
    # params_to_full 已移除。
    # 替代：外部自行取出参数的 x,y 通道和 r 通道：
    #         point_xy = params[:, [s, s+1]]            # 每控制点
    #         full_xy  = CoarseToFine.transform_point(point_xy, level.transform.point_affine)
    #         full_r   = CoarseToFine.transform_radius(params[:, 9], level.transform.r_scale)
    #       颜色/透明度等非几何参数不变。

    # ------------------------------------------------------------------ #
    # 迭代接口（后续阶段用；本阶段不调用）
    # ------------------------------------------------------------------ #
    def iter_levels(self):
        """逐级迭代（粗→细），yield Level。"""
        return iter(self.pyramid)

    def __iter__(self):
        return self.iter_levels()

    def __len__(self) -> int:
        return len(self.pyramid)

    # ------------------------------------------------------------------ #
    # 内存估算（不持有副本占用，只算张量显存/内存）
    # ------------------------------------------------------------------ #
    def memory_bytes(self) -> int:
        """金字塔所有切片张量占的字节数（float32）。

        注意：这是"若把整个金字塔同时存住"的开销。实际 coarse-to-fine 流程是逐级
        处理，常只持当前级；但本类为"优化空间"性质，各级切片需可重复访问（笔画优化
        多 epoch 复用同一切片），故整体常驻——故此项即常驻开销。
        """
        total = 0
        for lvl in self.pyramid:
            total += lvl.image.numel() * lvl.image.element_size()
        return total

    # ------------------------------------------------------------------ #
    # 诊断打印（数值，不读图）
    # ------------------------------------------------------------------ #
    def summary_str(self) -> str:
        lines = [
            f"target: {tuple(self.target.shape)}  cap={self.cap}  "
            f"levels={self.levels}",
            f"{'grid':>4} {'tiles':>5} {'patch_hw':>10} {'r_scale':>9} {'mem(KB)':>8}",
        ]
        for lvl in self.pyramid:
            # 同级等分：patch_hw / r_scale 取第 0 片代表（同级一致）
            mem_kb = lvl.image.numel() * lvl.image.element_size() / 1024.0
            lines.append(
                f"{lvl.grid_n:>4} {lvl.n_tiles:>5} {str(lvl.patch_hw):>10} "
                f"{float(lvl.transform.r_scale[0]):>9.4f} {mem_kb:>8.1f}"
            )
        lines.append(f"pyramid total mem = {self.memory_bytes()/1024.0:.1f} KB")
        return "\n".join(lines)


# ========================================================================== #
# 测试 main：切片金字塔可视化（plt 弹窗，不落盘，不读图入上下文）
# ========================================================================== #
def main():
    import matplotlib.pyplot as plt

    import device as dev
    import image_input
    from image_input import img_input

    print("=" * 64)
    print("coarse_to_fine — 子画布金字塔测试（第一阶段：切图/降采样/Transform）")
    print("=" * 64)

    img_input.load()
    print(f"[device] {dev.device}")
    print(f"[target] {tuple(img_input.target.shape)}  "
          f"[sum] {float(img_input.target.sum()):.3e}")

    c2f = CoarseToFine(img_input.target)
    print(f"[config] n_target={c2f.n_target}  factor={c2f.factor}  cap={c2f.cap}  "
          f"levels={c2f.levels}")

    print("-" * 64)
    print(c2f.summary_str())
    print("-" * 64)
    for lvl in c2f.pyramid:
        print(f"  grid={lvl.grid_n:>2} tiles={lvl.n_tiles} patch_hw={lvl.patch_hw} "
              f"r_scale={float(lvl.transform.r_scale[0]):.4f}")
        print(f"        point_affine[0] = {lvl.transform.point_affine[0].flatten().tolist()}")

    # ---- plt 可视化：各级切片金字塔拼图（弹窗，不落盘）----
    n_levels = len(c2f.pyramid)
    fig = plt.figure(figsize=(14, 3.2 * n_levels), dpi=140)
    for li, lvl in enumerate(c2f.pyramid):
        g = lvl.grid_n
        n_tiles = lvl.n_tiles
        per_row = min(n_tiles, 16)          # 每行最多 16 片，超出截断（细级片太多看不清）
        gs = fig.add_gridspec(
            1, per_row,
            left=0.05, right=0.98,
            bottom=1 - (li + 1) / n_levels + 0.01,
            top=1 - li / n_levels - 0.02,
        )
        fig.text(0.01, 1 - (li + 0.5) / n_levels,
                 f"g={g}\n{g}x{g}\n{n_tiles}",
                 va="center", ha="left", fontsize=9)
        for ti in range(per_row):
            ax = fig.add_subplot(gs[0, ti])
            img = lvl.image[ti, :3].permute(1, 2, 0).cpu().numpy()   # (h,w,3)
            ax.imshow(img, interpolation="nearest")
            ax.set_title(
                f"tile[{ti}]\n"
                f"{lvl.patch_hw[0]}x{lvl.patch_hw[1]}\n"
                f"r={float(lvl.transform.r_scale[ti]):.3f}",
                fontsize=7,
            )
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f"coarse-to-fine tile pyramid  target={tuple(img_input.target.shape)}  "
        f"n_target={c2f.n_target} factor={c2f.factor} cap={c2f.cap}",
        fontsize=12,
    )
    plt.show()


if __name__ == "__main__":
    main()
