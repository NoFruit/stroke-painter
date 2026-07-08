"""Stamp 采样与渲染的可微工具。

实现两条核心机制：

1. **Sigmoid 软计数** —— stamp 数量 N 由曲线长度 L 与间隔 d 的比值软化为
   ``N = Σ_{k=1}^{K} σ(α·(L/d - k))``，每个候选 stamp k 的激活权重
   ``w_k = σ(α·(L/d - k))`` 即该求和项。对 L（→几何）和 d（→半径）可微。

2. **Soft-disk 覆盖渲染** —— 每个 stamp 是一个 soft disk（相对半径归一化的
   sigmoid），所有 stamp 的覆盖做概率并集 ``M = 1 - ∏_k (1 - w_k·disk_k)``，
   即"像素落入任一 stamp"的 soft 版本。对 stamp 中心（→Bézier 参数）、半径、
   权重均可微。
"""

import torch
from torch import Tensor


def soft_stamp_count(L: Tensor, d: Tensor, K: int, alpha: float) -> Tensor:
    """Sigmoid 软计数：返回每个候选 stamp 的激活权重。

    ``w_k = σ(α · (L/d - k))``，k = 1..K。L/d 为理想连续 stamp 数；
    k < L/d 时 w_k → 1，k > L/d 时 w_k → 0，过渡处软化为分数。

    Args:
        L: ``(B,)`` 曲线长度。
        d: ``(B,)`` 或 ``(B, 1)`` stamp 间隔。
        K: 候选 stamp 上界。
        alpha: sigmoid 陡度。

    Returns:
        ``(B, K)`` 权重。
    """
    k = torch.arange(1, K + 1, device=L.device, dtype=L.dtype)  # (K,)
    ratio = (L / d.reshape(-1))[:, None]  # (B, 1)
    return torch.sigmoid(alpha * (ratio - k[None, :]))  # (B, K)


def render_coverage(
    stamp_centers: Tensor,
    radius_px: Tensor,
    weights: Tensor,
    grid_coords: Tensor,
    disk_softness: float,
) -> Tensor:
    """Soft-disk 并集覆盖。

    每个 stamp k 在像素 x 处的覆盖为
    ``m_k = w_k · σ(γ · (1 - ||x - x_k|| / r))``（相对半径归一化的 soft disk，
    γ 为无量纲边缘陡度，与分辨率无关）。像素总覆盖取概率并集：

    ``M(x) = 1 - ∏_k (1 - m_k)``

    Args:
        stamp_centers: ``(B, K, 2)`` stamp 中心（像素坐标）。
        radius_px: 像素半径，``(B, 1)`` 或 ``(B, K)``。
        weights: ``(B, K)`` 软计数激活权重。
        grid_coords: ``(H, W, 2)`` 像素中心坐标 [x, y]。
        disk_softness: γ，无量纲边缘陡度。

    Returns:
        ``(B, H, W)`` 覆盖率 ∈ [0, 1]。

    Note:
        中间张量形状为 ``(B, K, H, W)``；大 patch × 大 K 时显存随 B·K·H·W 增长，
        调用方据此选择 patch 大小或分批。
    """
    diff = stamp_centers[:, :, None, None, :] - grid_coords[None, None, :, :, :]  # (B,K,H,W,2)
    dist = torch.linalg.norm(diff, dim=-1)  # (B, K, H, W)

    r = radius_px
    if r.dim() == 2:  # (B, 1) 或 (B, K)
        r = r[:, :, None, None]
    elif r.dim() == 1:
        r = r[:, None, None, None]

    disk = torch.sigmoid(disk_softness * (1.0 - dist / r))  # (B, K, H, W)
    m = weights[:, :, None, None] * disk  # (B, K, H, W)
    coverage = 1.0 - torch.prod(1.0 - m, dim=1)  # (B, H, W)
    return coverage


def render_coverage_obb(
    stamp_centers: Tensor,
    tangents: Tensor,
    half_extent_px: Tensor,
    weights: Tensor,
    grid_coords: Tensor,
    softness: float,
) -> Tensor:
    """Soft-OBB（定向方框）并集覆盖。

    每个 stamp k 是一个以中心 x_k 为心、两轴半长均为 r（即边长 2r 的正方形）、
    一轴朝向曲线切线 u_k 的定向包围盒（OBB）。像素 x 到该 OBB 的有符号距离::

        la = (x - x_k)·u_k      lb = (x - x_k)·v_k        # 局部坐标，v_k ⊥ u_k
        qx = |la| - r           qy = |lb| - r
        outside = sqrt(max(qx,0)² + max(qy,0)² + 1e-12)
        inside  = clamp(max(qx,qy), max=0)
        sdf = outside + inside                                # <0 内，=0 边界，>0 外

    soft-box 取 ``box_k = σ(-γ · sdf / r)``：sdf=0（半轴 r 的方框边界）处 → 0.5，
    内部 → 1，外部 → 0。该形式与 soft-disk 的 ``σ(γ·(1 - ‖x-x_k‖/r))`` 同构
    （后者即 ``σ(-γ · sdf_disk / r)``，``sdf_disk = ‖x-x_k‖ - r``），故 r 在两刷
    中扮演相同的"可见特征尺度"角色——disk 为半径，方刷为半轴长。

    像素总覆盖取概率并集::

        m_k(x) = w_k · box_k(x)
        M(x)   = 1 - ∏_k (1 - m_k(x))

    Args:
        stamp_centers: ``(B, K, 2)`` stamp 中心（像素坐标）。
        tangents: ``(B, K, 2)`` 每个 stamp 处的**单位切线** u_k（调用方归一化）。
        half_extent_px: 半轴 r（两轴相同），``(B, 1)`` 或 ``(B, K)``。
        weights: ``(B, K)`` 软计数激活权重。
        grid_coords: ``(H, W, 2)`` 像素中心坐标 [x, y]。
        softness: γ，无量纲边缘陡度。

    Returns:
        ``(B, H, W)`` 覆盖率 ∈ [0, 1]。

    Note:
        中间张量形状为 ``(B, K, H, W)`` 且约 8 个（la/lb/qx/qy/outside/inside/sdf/box），
        比 soft-disk 多；大 patch × 大 K 时显存随 B·K·H·W 增长更明显，调用方据此
        选择 patch 大小或分批。``outside`` 的 ``+1e-12`` 是必要的：方框内部是正面积
        区上恒为 0（不像 disk 仅在中心测度零点处为 0），无 eps 则 sqrt(0) 反传为 inf/NaN。
    """
    diff = stamp_centers[:, :, None, None, :] - grid_coords[None, None, :, :, :]  # (B,K,H,W,2)

    # 切线 u_k 与垂直方向 v_k = (-u_y, u_x)（box 两轴半长相等，v 的符号无关）。
    u = tangents[:, :, None, None, :]                          # (B,K,1,1,2)
    v = torch.stack([-tangents[..., 1], tangents[..., 0]], dim=-1)  # (B,K,2)
    v = v[:, :, None, None, :]                                 # (B,K,1,1,2)
    la = (diff * u).sum(-1)                                    # (B,K,H,W)
    lb = (diff * v).sum(-1)                                    # (B,K,H,W)

    r = half_extent_px
    if r.dim() == 2:  # (B,1) 或 (B,K)
        r = r[:, :, None, None]
    elif r.dim() == 1:
        r = r[:, None, None, None]

    qx = la.abs() - r                                          # (B,K,H,W)
    qy = lb.abs() - r
    outside = torch.sqrt(qx.clamp(min=0.0).pow(2) + qy.clamp(min=0.0).pow(2) + 1e-12)
    inside = torch.clamp(torch.maximum(qx, qy), max=0.0)
    sdf = outside + inside                                     # 有符号距离
    box = torch.sigmoid(-softness * sdf / r)                   # (B,K,H,W) soft-OBB

    m = weights[:, :, None, None] * box                        # (B,K,H,W)
    coverage = 1.0 - torch.prod(1.0 - m, dim=1)                # (B,H,W)
    return coverage


def cut_endcaps(
    coverage: Tensor,
    grid_coords: Tensor,
    P0: Tensor,
    P2: Tensor,
    u0: Tensor,
    u2: Tensor,
    sharpness: float,
) -> Tensor:
    """半平面 sigmoid 端点切割：将圆刷覆盖的圆头切为方头。

    对起点 P0，以切线 u0 方向为"前方"，切除后方（半圆头延伸区）；对终点 P2，
    以切线 u2 方向为"前方"，切除前方（半圆头延伸区）。切割用 sigmoid 软过渡，
    可微、O(HW)、过渡区宽度 ≈ 1/sharpness 像素。

    原理：::

        cut_start(x) = σ(β · (x − P0) · u0)     # 保留前方，切除后方
        cut_end(x)   = σ(β · (P2 − x) · u2)     # 保留前方，切除后方
        M_square(x)  = M_round(x) · cut_start(x) · cut_end(x)

    Args:
        coverage: 圆刷覆盖 ``(B, H, W)``，由 :func:`render_coverage` 产出。
        grid_coords: 像素中心坐标 ``(H, W, 2)``，通道顺序 [x, y]。
        P0: 起点像素坐标 ``(B, 2)``。
        P2: 终点像素坐标 ``(B, 2)``。
        u0: 起点处单位切线 ``(B, 2)``，方向指向曲线内部。
        u2: 终点处单位切线 ``(B, 2)``，方向指向曲线内部。
        sharpness: sigmoid 陡度 β（无量纲）。值越大切割越锐利，过渡区越窄。

    Returns:
        ``(B, H, W)`` 方头覆盖率 ∈ [0, 1]。
    """
    # cut_start: 保留 P0 前方 (沿 u0)，切除后方
    dot_start = ((grid_coords[None, ...] - P0[:, None, None, :])
                 * u0[:, None, None, :]).sum(-1)  # (B, H, W)
    cut_start = torch.sigmoid(sharpness * dot_start)

    # cut_end: 保留 P2 前方 (沿 u2)，切除后方
    dot_end = ((P2[:, None, None, :] - grid_coords[None, ...])
               * u2[:, None, None, :]).sum(-1)  # (B, H, W)
    cut_end = torch.sigmoid(sharpness * dot_end)

    return coverage * cut_start * cut_end


def pixel_grid(H: int, W: int, device, dtype) -> Tensor:
    """像素中心坐标网格 (H, W, 2)，通道顺序 [x, y]。

    所有笔刷共享的像素网格生成，避免在每个实现中重复定义。
    """
    ys = torch.arange(H, device=device, dtype=dtype) + 0.5
    xs = torch.arange(W, device=device, dtype=dtype) + 0.5
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1)  # (H, W, 2)
