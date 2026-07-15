"""PointCloudOTLoss — 点云版 OT 损失原语（geomloss 点云路径，无状态）。

套壳 ``geomloss.SamplesLoss(loss="sinkhorn", backend="online")``：把图像压成
**固定数量 N 的点云**（权重 = 降采样后的像素质量，坐标 = 落在 ``[0,1)^2`` 的粗
网格点），再做 debiased Sinkhorn 散度。对应 BofP 的 ``L_OT`` 项。

- grid 版吃整图，但 0.3.1 grid 求解器为坏桩（详见
  ``docs/Loss_OT_geomloss_refs/geomloss_ref.md``），不可用。
- **backend=online（keops Genred 在线 LSE，不实体化 N×N）**。key_averages 实测：
  ot forward 的 98% CUDA 时间在 ``GenredAutograd``（pykeops 编译 kernel），喂给
  sinkhorn 多步迭代的 softmin。
- **backend 选择**：同 N 下 online 比 tensorized 快 ~2.5x
  （N=576: online 50s vs tensorized 66s）。tensorized 实体化 N×N 代价矩阵 + exp 算术
  开销大，仅在 N 大到 OOM 时被迫用；同 N 下它就是更慢。

点云化策略（固定 N，且 N 必须是完全平方数 → s×s 网格）
------------------------------------------------------
图像 → 质量分布 (B,1,H,W) → ``F.adaptive_avg_pool2d`` 到 (B,1,s,s) → 展平为
N=s*s 个点。**权重 = 池化后质量（带梯度，可 backward 到笔刷参数）；坐标 = s×s
粗网格的固定像素中心（与图像内容无关，detach）**。这样 OT 只在"同一粗网格上的
两份质量分布"间求解，完全可导、确定、零采样噪声。N 取 1024 (s=32) 时
tensorized 代价矩阵 1024²≈1M，远低于 ~5000² 的 tensorized 上限。

原语语义（与 L1/Grad/Area 等同级，统一契约）
---------------------------------------------
- **无状态**：``self`` 不持 cache、不缓存 target。``forward`` 是忠实单次计算，
  输入→输出，结果不被任何后续调用影响。组合 cache 由上层 :mod:`loss_space` 统一持有。
- :meth:`forward` 返回 ``(scalar, info)``：scalar 可 backward 到 pred；info 带出
  可视化所需中间量（OT 残差 viz 的 a/b 点云权重），零重算。
- target 由调用方每次传入（loss_space 跨步持有 target）。

设计要点
--------
- **维度范式**：计算空间统一四维 ``(B,3,H,W)``。输入 ``pred_img`` / ``target_img``
  均为 ``(B,3,H,W)``（B≥1，批处理：一批切片一对一算 OT）；点云化时把每张图的
  质量 ``(B,1,s,s)`` 展平成 ``(B,N)`` 权重 + ``(B,N,2)`` 坐标喂给 SamplesLoss
  （geomloss 点云 API 的约定形状：权重 ``(B,N)``、坐标 ``(B,N,D)``，逐 batch 求 OT）。
  质量归一化、坐标广播、loss 求平均**全部逐 batch 独立**，B 间不串。
- 参数全部硬编码在 ``__init__``（p / blur / reach / debias / scaling /
  n_points / mass_mode / backend），无命令行 / 配置文件。
- :meth:`to_points` —— 图像 → (权重, 坐标) 点云（计算用结构，公开，供上层复用）。
- :meth:`visual` —— 把一次 forward 产出的 info 整理成 viz 字典（纯函数，不读 self）。
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .loss_base import LossBase


class PointCloudOTLoss(LossBase):
    """点云版 debiased Sinkhorn OT 损失原语（online 后端 / keops Genred，无状态）。

        输入约定（按需：OT 从图像派生质量分布，吃 RGB；target 仅 RGB 无 alpha，
        故 pred/target 对称都从 RGB 派生质量）：
            pred_img   : (B,3,H,W) float [0,1]，当前画布状态（带梯度，合成后的 RGB canvas）。
                B≥1；B>1 即批处理——一批切片一对一算 OT，B 间独立。
        target_img : (B,3,H,W) float [0,1]，目标图（与 pred 同 B）。

    输出（统一契约 ``(scalar, info)``）：
        forward(...)  → (标量 loss, info dict)，scalar = 各 batch OT 的均值，
                        可 backward 到 pred。info 含 viz 中间量：a / b 点云权重
                        (B,N) + side，供 :meth:`visual`。
    """

    def __init__(self):
        import geomloss  # 延迟导入：仅 PointCloudOTLoss 实例化时触发，L1/Grad/Area 不被拖累

        # ---- 死代码参数（硬编码在 init）----
        self.p = 2                 # 地面代价指数；p=2 → ½|x-y|²（Wasserstein-2）
        self.blur = 0.05           # Gibbs 核带宽 σ，ε = blur^p。坐标在 [0,1)^2 时合理
        self.reach = 0.3           # unbalanced 强度 ρ = reach^p；非 None 容许两边质量
                                    # 不匹配（笔触覆盖不足时更鲁棒）。None = 平衡 OT
        self.debias = True         # 去偏 Sinkhorn 散度 S_ε（正定，a=b⇒loss=0）
        self.scaling = 0.5         # ε-scaling 退火比，[0.5,1)
        self.n_points = 576       # 固定点云数量 N；必须为完全平方数 -> s×s 粗网格。
                                   # N 是 OT 的核心杠杆 ：
                                   # online 下 N 4096->576，loss.ot -77%、wall -39%。
                                   # online 无 OOM 墙（不实体化 N×N），N 可继续往下探，但带画质成本。
        self.mass_mode = "luminance"  # 图像 → 质量的取法（见 to_points）
        self.mass_eps = 1e-6       # 质量地板，防 log(0)
        # keops Genred 在线 LSE（不实体化 N×N）；key_averages 实测占 ot 98%。
        self.backend = "online"
        # "tensorized" "online" "multiscale"
        # backend 选择见 docstring 顶部实测结论：同 N 下 online >> tensorized（N=576: 50s vs 66s），
        # tensorized 实体化 N×N+exp 开销大，仅 OOM 时被迫用。勿切 tensorized。
        self.chunk_b = 128          # 批 OT 串行分块大小（仅 tensorized 后端用）。tensorized 实体化
                                    # 代价矩阵 (B,N,N)，B·N²·4 字节随 B 爆炸；B 超显存时按 chunk_b
                                    # 分块串行喂 SamplesLoss（各 batch 独立，分块求和再除 B == 整批
                                    # 均值，语义不变，仅时间换显存）。online 不实体化 N×N，无 OOM 墙，
                                    # chunk_b 无意义（不触发分块）。

        # ---- 派生：s = sqrt(N)，校验完全平方 ----
        s = int(round(self.n_points ** 0.5))
        if s * s != self.n_points:
            raise ValueError(
                f"n_points 必须为完全平方数（s×s 粗网格），got {self.n_points}"
            )
        self._side = s             # 粗网格边长 s

        # ---- SamplesLoss 实例（参数冻结，复用；其本身无待训练参数）----
        self._loss_fn = geomloss.SamplesLoss(
            loss="sinkhorn",
            p=self.p,
            blur=self.blur,
            reach=self.reach,
            scaling=self.scaling,
            debias=self.debias,
            backend=self.backend,
        )

        # ---- 固定坐标网格 (N,2) detach，与图像内容无关 ----
        # 惰性建：第一次 forward 时用输入张量的 device 建，缓存到 self._coords。
        # 存无 batch 维的 (N,2)；按需 expand 到 (B,N,2)，避免为每个 B 预存副本。
        self._coords: Tensor = None

    # ------------------------------------------------------------------ #
    # 内部：固定坐标网格（惰性建，device 跟随输入张量）
    # ------------------------------------------------------------------ #
    def _get_coords(self, device: torch.device, B: int) -> Tensor:
        """返回 (B,N,2) 坐标网格。惰性建 + 按 device 缓存。

        第一次调用时用 ``device`` 建网格并缓存到 ``self._coords``；
        后续若 device 不变则复用，device 变了则重建（跨 device 切换安全）。
        """
        s = self._side
        if self._coords is None or self._coords.device != device:
            rows, cols = torch.meshgrid(
                torch.arange(s, device=device),
                torch.arange(s, device=device),
                indexing="ij",
            )
            coords = torch.stack([(cols + 0.5) / s, (rows + 0.5) / s], dim=-1)  # (s,s,2)
            self._coords = coords.reshape(s * s, 2).detach()  # (N,2)
        return self._coords[None].expand(B, -1, -1).contiguous()  # (B,N,2)

    # ------------------------------------------------------------------ #
    # 图像 → 计算用结构
    # ------------------------------------------------------------------ #
    def _to_mass(self, img: Tensor) -> Tensor:
        """(B,3,H,W) float [0,1] → (B,1,H,W) 归一化质量分布（逐 batch 独立归一）。

        输入为 RGB（无 alpha 通道），质量按 mass_mode 从 RGB 派生
        （luminance/inv_luminance/mean/inv_mean）。
        质量归一化**逐 batch**：每张图各自归一为概率分布，B 间不串。
        全零 batch 退化为均匀分布（如 inv_luminance 遇到纯白空白画布）。
        """
        img = img.float()
        rgb = img[:, :3]                       # (B,3,H,W)

        if self.mass_mode == "luminance":
            m = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
        elif self.mass_mode == "inv_luminance":
            m = 1.0 - (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2])
        elif self.mass_mode == "mean":
            m = rgb.mean(1)
        elif self.mass_mode == "inv_mean":
            m = 1.0 - rgb.mean(1)
        else:
            raise ValueError(f"未知 mass_mode: {self.mass_mode}（仅支持 "
                            f"luminance/inv_luminance/mean/inv_mean）")
        m = m[:, None]                         # (B,1,H,W)

        m = m.clamp(min=0)
        B, _, H, W = m.shape
        # 逐 batch 归一化：对每张图的空间维求和（keepdim），B 间独立。
        s = m.flatten(1).sum(1, keepdim=True)[:, :, None, None]   # (B,1,1,1)
        zero_b = s.detach().squeeze() <= 0     # 全零 batch 掩码（B→标量时仍可用）
        if zero_b.any():
            uniform = torch.full_like(m, 1.0 / (H * W))
            m = torch.where(zero_b.view(B, 1, 1, 1), uniform, m / (s + 1e-12))
        else:
            m = m / s
        m = m.clamp(min=self.mass_eps)
        m = m / m.flatten(1).sum(1, keepdim=True)[:, :, None, None]   # 逐 batch 再归一
        return m                               # (B,1,H,W)

    def to_points(self, img: Tensor) -> Tuple[Tensor, Tensor]:
        """(B,3,H,W) → (权重 (B,N), 坐标 (B,N,2)) 点云（逐 batch 独立归一）。

        质量 (B,1,H,W) → adaptive_avg_pool 到 (B,1,s,s) → 展平 (B,N) 权重（带
        梯度）；坐标取固定 s×s 网格 expand 到 (B,N,2)（detach）。权重逐 batch 再
        归一化 + eps 地板。

        公开：上层 loss_space 可对 target 调用一次、跨步复用其 (b, y)。
        """
        m = self._to_mass(img)                          # (B,1,H,W) 带梯度
        B = m.shape[0]
        s = self._side
        m = F.adaptive_avg_pool2d(m, (s, s))            # (B,1,s,s) 带梯度
        a = m.reshape(B, s * s)                         # (B,N) 权重，带梯度
        a = a.clamp(min=self.mass_eps)
        a = a / a.sum(1, keepdim=True)                  # 逐 batch 归一
        return a, self._get_coords(img.device, B)       # (B,N) , (B,N,2) detach

    # ------------------------------------------------------------------ #
    # 常用 loss（标量，可 backward）+ info
    # ------------------------------------------------------------------ #
    def forward(self, pred_img: Tensor, target_img: Tensor) -> Tuple[Tensor, dict]:
        """计算 pred 画布与 target 之间的 debiased Sinkhorn OT 散度（批处理）。

        无状态：target 每次由调用方传入（loss_space 跨步持有 target）。
        批处理：pred/target 同 B，逐 batch 求 OT，再对 B 取均值得标量。

        Args:
            pred_img: (B,3,H,W) 当前画布状态，带梯度（合成后的 RGB canvas）。
            target_img: (B,3,H,W) 目标图（内部 detach，与 pred 同 B）。

        Returns:
            (loss, info)：loss 标量（各 batch OT 均值），梯度可通到 ``pred_img``；
            info 带出 a/b 点云权重 (B,N)（detach）+ side，供 :meth:`visual`。
        """
        a, x = self.to_points(pred_img)                 # (B,N) 带梯度, (B,N,2)
        b, y = self.to_points(target_img.detach())      # (B,N) detach, (B,N,2)

        # 逐 batch OT 再均值。tensorized 后端代价矩阵 (B,N,N) 随 B·N² 爆炸；B 超过显存
        # 能装量时按 chunk_b 串行分块（各 batch OT 独立，分块求和/B == 整批均值，仅时间
        # 换显存）。chunk_b ≥ B 或 =0 → 一次并行不分块。
        B = a.shape[0]
        if self.chunk_b and self.chunk_b < B:
            acc = [self._loss_fn(a[i:j], x[i:j], b[i:j], y[i:j])
                   for i in range(0, B, self.chunk_b)
                   for j in (min(i + self.chunk_b, B),)]   # 各 (j-i,)
            loss = torch.cat(acc).mean()
        else:
            loss = self._loss_fn(a, x, b, y).mean()     # (B,) → 标量

        info = {
            "a": a.detach(),        # (B,N) pred 点云权重
            "b": b.detach(),        # (B,N) target 点云权重
            "side": self._side,
        }
        return loss, info

    # ------------------------------------------------------------------ #
    # 可视组装（纯函数：吃 forward 的 info，不读 self 状态）
    # ------------------------------------------------------------------ #
    def visual(self, info: dict) -> Dict[str, Tensor]:
        """把一次 :meth:`forward` 的 info 整理成 viz 字典（纯函数，不读 self）。

        点云权重 reshape 回 (B,1,s,s) 质量热图——与 grid 版同形，直接复用
        ``viewer.show_ot_visual``。残差 a-b 为**主可视**：正值=画布过量(该减)，
        负值=欠量(该补笔)。

        Returns:
            dict，键值均为 (B,1,s,s) 张量（已 detach，四维标准；由 viewer 导出边界
            降维展示）：
                - "pred_mass"    : 当前画布点云权重 reshape 回 (B,1,s,s)
                - "target_mass"  : 目标点云权重 reshape 回 (B,1,s,s)
                - "residual"     : a - b，**主可视**
        """
        s = info["side"]
        a = info["a"].detach()                          # (B,N)
        b = info["b"].detach()                          # (B,N)
        B = a.shape[0]
        return {
            "pred_mass": a.reshape(B, 1, s, s),         # (B,1,s,s)
            "target_mass": b.reshape(B, 1, s, s),       # (B,1,s,s)
            "residual": (a - b).reshape(B, 1, s, s),    # 主可视：过/欠画图
        }

    # ------------------------------------------------------------------ #
    # 设计注记
    # ------------------------------------------------------------------ #
    # 原语无状态：cache 与 target 缓存全部上移到 loss_space。本类只忠实算一次
    # (pred,target)→loss，并经 info 把 viz 中间量带出。组合、加权、暂存可视化快照、
    # target 跨步复用都由 loss_space 统一持有——单一 cache owner，不再每类各存一份。
