"""二次 Bézier 曲线的可微几何工具。

包含系数求解、曲线求值、导数、弧长（闭式解）以及"弧长 -> 参数 t"的反演
（基于细网格累积弧长 + 线性插值）。弧长采用 kurbo（Raph Levien）的稳健闭式 +
近直线 Gauss-Legendre 回退；反演仍为数值法（二次弧长反函数无初等闭式）。
所有运算对控制点可微，梯度可经 autograd 反传至 Bézier 几何参数。

二次 Bézier 曲线 ``B(t) = (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2`` 可展开为解析
多项式 ``B(t) = a t^2 + b t + c``，其中::

    a = P0 - 2 P1 + P2      （二次项）
    b = 2 (P1 - P0)         （一次项）
    c = P0                  （常数项）

:func:`bezier_eval`、:func:`bezier_derivative`、:func:`arc_length_to_t` 均直接基于
系数 (a, b, c) 计算，省去重复展开；:func:`bezier_coefficients` 完成控制点 -> 系数的可微映射。
"""

import math

import torch
from torch import Tensor


def bezier_coefficients(P0: Tensor, P1: Tensor, P2: Tensor):
    """求解二次 Bézier 的解析式系数。

    将 ``B(t) = (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2`` 展开为
    ``B(t) = a t^2 + b t + c``，返回三个系数向量::

        a = P0 - 2 P1 + P2      （二次项）
        b = 2 (P1 - P0)         （一次项）
        c = P0                  （常数项）

    三系数均为控制点的线性组合，故对 P0/P1/P2 可微，梯度可经 autograd
    反传至 Bézier 几何参数。

    Args:
        P0, P1, P2: 控制点，形状 ``(B, 2)``（或任意可广播形状）。

    Returns:
        (a, b, c)：三个与控制点同形状的系数张量。
    """
    a = P0 - 2.0 * P1 + P2
    b = 2.0 * (P1 - P0)
    c = P0
    return a, b, c


def bezier_eval(a: Tensor, b: Tensor, c: Tensor, t: Tensor) -> Tensor:
    """二次 Bézier 求值 B(t) = a t^2 + b t + c。

    系数 (a, b, c) 由控制点经 :func:`bezier_coefficients` 求得，等价于
    ``(1-t)^2 P0 + 2(1-t)t P1 + t^2 P2``。

    Args:
        a, b, c: 解析式系数，形状 ``(B, 2)``。
        t: 曲线参数，形状 ``(B, K)``。

    Returns:
        曲线点，形状 ``(B, K, 2)``。
    """
    a = a[:, None, :]
    b = b[:, None, :]
    c = c[:, None, :]
    t = t.unsqueeze(-1)
    return a * (t * t) + b * t + c


def bezier_derivative(a: Tensor, b: Tensor, c: Tensor, t: Tensor) -> Tensor:
    """B'(t) = 2 a t + b。

    系数 (a, b, c) 由控制点经 :func:`bezier_coefficients` 求得，等价于
    ``2[(1-t)(P1-P0) + t(P2-P1)]``。``c`` 不参与求导，仅保持与
    :func:`bezier_eval` 一致的 (a, b, c) 签名。

    Args:
        a, b, c: 解析式系数，形状 ``(B, 2)``。
        t: ``(B, K)``。

    Returns:
        导数向量，形状 ``(B, K, 2)``。
    """
    a = a[:, None, :]
    b = b[:, None, :]
    t = t.unsqueeze(-1)
    return 2.0 * a * t + b


def bezier_length(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """曲线总弧长 L = ∫_0^1 ||B'(t)|| dt 的闭式解。

    二次 Bézier 弧长有初等闭式（三次则无，为椭圆积分）。采用 kurbo
    (Raph Levien) 的稳健实现：闭式公式 + 近直线时回退 3 点 Gauss-Legendre
    求积，规避 ``||a||/||b|| -> 0``（近乎直线 / 退化）时的浮点抵消
    （参考 kurbo ``QuadBez::arclen``，与 Loria 的 "simple formula" 等价）。

    由 ``B'(t) = 2a t + b``，``||B'(t)||² = 4(a·a)t² + 4(a·b)t + (b·b)``。
    设 ``A=||a||², B=a·b, C=||b||²/4``，记 ``sabc=√(A+B+C)=||B'(1)||/2``、
    ``cb=||b||=||B'(0)||``，闭式为::

        v0 = ¼·(B/A)·(2·sabc − cb) + sabc
        L  = v0 + ¼·A^(-3/2)·det²(a,b)·ln( ((2A+B)/||a|| + 2·sabc) / (B/||a|| + cb) )

    其中 ``det²(a,b)=‖a‖²‖b‖²−(a·b)²``（= 4AC−B²，由 2D 叉积直接构造，稳定）。
    ``c=P0`` 不参与（弧长平移不变），仅保持与 :func:`bezier_eval` /
    :func:`bezier_derivative` 一致的 (a,b,c) 签名。

    Args:
        a, b, c: 解析式系数，形状 ``(B, 2)``。

    Returns:
        ``(B,)`` 弧长。

    Note:
        - 近直线（``||a||² < 5e-4·||b||²/4``，即 ``||a||/||b|| < ~1.1e-2``）回退
          3 点 Gauss-Legendre 求积（规避闭式中的减法抵消）。
        - 尖折（``B'(t)`` 在 [0,1] 过零，cusp，``B/||a||+||b||`` 极小）跳过 log 项。
        - 全退化（a=b=0）经近直线分支返回 0。各分支均经 clamp 保证有限，反向不生 NaN。
    """
    aa = (a * a).sum(-1)                              # (B,) ||a||²
    ab = (a * b).sum(-1)                              # (B,) a·b
    bb = (b * b).sum(-1)                              # (B,) ||b||²
    A, B, C = aa, ab, bb * 0.25
    eps = 1e-18

    # --- 3 点 Gauss-Legendre（近直线回退）：L ≈ Σ w_i ||B'(t_i)||, B'(t)=2a t+b
    s = math.sqrt(3.0 / 5.0)
    t1 = 0.5 - 0.5 * s                               # 0.1127016654
    t3 = 0.5 + 0.5 * s                               # 0.8872983346
    we = 5.0 / 18.0
    wm = 8.0 / 18.0
    n1 = (2.0 * t1 * a + b).norm(dim=-1)
    nm = (a + b).norm(dim=-1)                        # B'(0.5) = a + b
    n3 = (2.0 * t3 * a + b).norm(dim=-1)
    L_gauss = we * n1 + wm * nm + we * n3

    # --- 闭式（kurbo）。A_safe / +eps 等保证未选中分支亦有限（反向无 NaN）。
    A_safe = A.clamp(min=eps)
    inv_a = 1.0 / A_safe.sqrt()                      # 1/||a||（退化处被垫高）
    sabc = 0.5 * (2.0 * a + b).norm(dim=-1)           # ||B'(1)||/2，由范数保证 ≥0
    cb = b.norm(dim=-1)                              # ||b|| = ||B'(0)||
    ba_c2 = B * inv_a + cb                           # B/||a|| + ||b||
    v0 = 0.25 * inv_a * inv_a * B * (2.0 * sabc - cb) + sabc
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    det2 = cross * cross                             # = ‖a‖²‖b‖²−(a·b)² ≥ 0
    log_arg = ((2.0 * A + B) * inv_a + 2.0 * sabc) / (ba_c2 + eps)
    log_term = 0.25 * (inv_a * inv_a * inv_a) * det2 * torch.log(log_arg.clamp(min=eps))
    kink = (ba_c2 * inv_a) < 1e-13                   # cusp：B'(t) 过零
    L_closed = torch.where(kink, v0, v0 + log_term)

    near_straight = (A < (5e-4 * C + eps))                                            # 含 A≈0（全退化）回退 Gauss-Legendre
    return torch.where(near_straight, L_gauss, L_closed)


def arc_length_to_t(a: Tensor, b: Tensor, c: Tensor, s_targets: Tensor, M: int) -> Tensor:
    """弧长 -> 参数 t 的反演。

    在 [0,1] 上取 M 点细网格，trapezoid 累积弧长 ``s_grid``（``s_grid[:,0]=0``、
    ``s_grid[:,-1]=L``）；对每个目标弧长 ``s_targets`` 做 searchsorted 定位段，
    再在段内线性插值得到 t。段索引为整数（直通），插值对 s_grid 与 s_targets
    可微，因此梯度可经弧长反传至系数（->控制点）。超出 [0, L] 的目标被映射到
    端点 t=1（其 stamp 权重由软计数决定，通常 ~0）。

    反演为数值法（二次弧长反函数无初等闭式）；总弧长请用 :func:`bezier_length`
    的闭式解，二者各自独立（本函数的梯形表总长与闭式差 ~1/M²）。

    Args:
        a, b, c: 解析式系数，形状 ``(B, 2)``。``c`` 不参与（反演平移不变），
            仅保持 (a,b,c) 签名一致。
        s_targets: ``(B, K)`` 目标弧长。
        M: 累积弧长网格分辨率。

    Returns:
        ``(B, K)`` 对应的参数 t。
    """
    B = a.shape[0]
    t_grid = torch.linspace(0.0, 1.0, M, device=a.device, dtype=a.dtype)
    d = bezier_derivative(a, b, c, t_grid.unsqueeze(0).expand(B, M))  # (B, M, 2)
    seg = torch.linalg.norm(d, dim=-1)                                  # (B, M) = ||B'(t)||
    dt = 1.0 / (M - 1)
    seg_len = 0.5 * (seg[:, :-1] + seg[:, 1:]) * dt                     # (B, M-1) 每段长度
    s_grid = torch.zeros(B, M, device=a.device, dtype=a.dtype)
    s_grid[:, 1:] = torch.cumsum(seg_len, dim=-1)

    idx = torch.searchsorted(s_grid, s_targets, right=True)             # (B, K)
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


def bezier_extrema(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """二次 Bézier 的极值点参数（用于 AABB 包围盒）。

    曲线分量的极值出现在 ``B'(t)`` 分量为零处。由 ``B'(t) = 2a t + b``，
    每个轴至多一个临界点::

        t_x = -b_x / (2 a_x),   t_y = -b_y / (2 a_y)

    等价于 Inigo Quilez / Pomax / kurbo 的 ``t = (x0-x1)/(x0-2x1+x2)``
    （因 ``a = P0-2P1+P2``、``b = 2(P1-P0)``，故 ``-b_axis/2 = x0-x1``、
    ``a_axis = x0-2x1+x2``）。

    返回每个轴的临界 ``t`` 并 **钳制到 [0, 1]**：超出 [0,1] 的临界点（该轴在
    [0,1] 上单调，极值在端点）与退化轴（``a_axis ≈ 0``，该轴线性无内部极值）
    均塌缩到端点，故调用方只需在候选集 ``t ∈ {0, 1, t_ext[:,0], t_ext[:,1]}``
    上求值并取分量 min/max 即得精确 AABB::

        t_cand = [0, 1, t_ext[:,0], t_ext[:,1]]      # (B, 4)
        pts = bezier_eval(a, b, c, t_cand)           # (B, 4, 2)
        aabb_min, aabb_max = pts.min(1), pts.max(1)   # 各 (B, 2)

    Args:
        a, b, c: 解析式系数，形状 ``(B, 2)``。``c`` 不参与（极值平移不变），
            仅保持 (a,b,c) 签名一致。

    Returns:
        ``(B, 2)``：每行 ``[t_x, t_y]``，均在 [0, 1]。

    Note:
        退化轴的分母用 1 垫高（避免除零），结果经 clamp 塌缩到端点，对 AABB 无害
        （线性轴的中点值夹在端点值之间，不参与 min/max）；全程无 NaN，反向安全，
        被钳制处梯度为 0（标准 clamp 行为）。
    """
    eps = 1e-12
    ax, ay = a[..., 0], a[..., 1]
    bx, by = b[..., 0], b[..., 1]
    # 退化轴分母置 1（避免 1/0），结果经 clamp 塌缩到端点，对 AABB 无害。
    ax_safe = torch.where(ax.abs() > eps, ax, torch.ones_like(ax))
    ay_safe = torch.where(ay.abs() > eps, ay, torch.ones_like(ay))
    tx = (-bx / (2.0 * ax_safe)).clamp(0.0, 1.0)
    ty = (-by / (2.0 * ay_safe)).clamp(0.0, 1.0)
    return torch.stack([tx, ty], dim=-1)               # (B, 2)
