"""二次 Bézier 曲线的可微几何工具。

包含曲线求值、导数、弧长（数值积分）以及"弧长 → 参数 t"的反演
（基于细网格累积弧长 + 线性插值）。所有运算对控制点可微，梯度可经
autograd 反传至 Bézier 几何参数。
"""

import torch
from torch import Tensor


def bezier_eval(P0: Tensor, P1: Tensor, P2: Tensor, t: Tensor) -> Tensor:
    """二次 Bézier 求值 B(t) = (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2。

    Args:
        P0, P1, P2: 控制点，形状 ``(B, 2)``。
        t: 曲线参数，形状 ``(B, K)``。

    Returns:
        曲线点，形状 ``(B, K, 2)``。
    """
    P0 = P0[:, None, :]
    P1 = P1[:, None, :]
    P2 = P2[:, None, :]
    t = t.unsqueeze(-1)
    one_t = 1.0 - t
    return (one_t * one_t) * P0 + (2.0 * one_t * t) * P1 + (t * t) * P2


def bezier_derivative(P0: Tensor, P1: Tensor, P2: Tensor, t: Tensor) -> Tensor:
    """B'(t) = 2[(1-t)(P1-P0) + t(P2-P1)]。

    Args:
        P0, P1, P2: ``(B, 2)``。
        t: ``(B, K)``。

    Returns:
        导数向量，形状 ``(B, K, 2)``。
    """
    a = (P1 - P0)[:, None, :]
    b = (P2 - P1)[:, None, :]
    t = t.unsqueeze(-1)
    return 2.0 * ((1.0 - t) * a + t * b)


def _cumulative_arc_length(P0: Tensor, P1: Tensor, P2: Tensor, M: int):
    """在 [0,1] 上均匀取 M 点，trapezoid 累积弧长。

    Returns:
        s_grid: ``(B, M)``，s_grid[:, 0] = 0，s_grid[:, -1] = L。
        t_grid: ``(M,)``，对应的均匀参数。
    """
    B = P0.shape[0]
    t_grid = torch.linspace(0.0, 1.0, M, device=P0.device, dtype=P0.dtype)
    d = bezier_derivative(P0, P1, P2, t_grid.unsqueeze(0).expand(B, M))  # (B, M, 2)
    seg = torch.linalg.norm(d, dim=-1)  # (B, M) = ||B'(t)||
    dt = 1.0 / (M - 1)
    seg_len = 0.5 * (seg[:, :-1] + seg[:, 1:]) * dt  # (B, M-1) 每段长度
    s_grid = torch.zeros(B, M, device=P0.device, dtype=P0.dtype)
    s_grid[:, 1:] = torch.cumsum(seg_len, dim=-1)
    return s_grid, t_grid


def bezier_length(P0: Tensor, P1: Tensor, P2: Tensor, M: int) -> Tensor:
    """曲线总弧长 L = ∫_0^1 ||B'(t)|| dt（trapezoid 数值积分）。

    Returns:
        ``(B,)``。
    """
    s_grid, _ = _cumulative_arc_length(P0, P1, P2, M)
    return s_grid[:, -1]


def arc_length_to_t(P0: Tensor, P1: Tensor, P2: Tensor, s_targets: Tensor, M: int) -> Tensor:
    """弧长 → 参数 t 的反演。

    在细网格累积弧长上对每个目标弧长 ``s_targets`` 做 searchsorted 定位段，
    再在段内线性插值得到 t。段索引为整数（直通），插值对 s_grid 与 s_targets
    可微，因此梯度可经弧长反传至 Bézier 控制点。超出 [0, L] 的目标被映射到
    端点 t=1（其 stamp 权重由软计数决定，通常 ~0）。

    Args:
        s_targets: ``(B, K)`` 目标弧长。

    Returns:
        ``(B, K)`` 对应的参数 t。
    """
    s_grid, t_grid = _cumulative_arc_length(P0, P1, P2, M)  # (B, M), (M,)
    idx = torch.searchsorted(s_grid, s_targets, right=True)  # (B, K)
    idx = idx.clamp(min=1, max=M - 1)
    i0 = idx - 1
    i1 = idx
    s0 = s_grid.gather(1, i0)
    s1 = s_grid.gather(1, i1)
    t0 = t_grid[i0]
    t1 = t_grid[i1]
    denom = s1 - s0 + 1e-12  # 退化段（零长曲线）数值保护
    frac = (s_targets - s0) / denom
    return t0 + frac * (t1 - t0)
