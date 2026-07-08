"""timing_probe.py — main 耗时归因探针（A 方向：不改源码，运行时 monkeypatch）。

潜入真实 ``main.main()``，给关键调用套计时壳（含 ``cuda.synchronize``），按 level ×
phase 聚合，跑完打印表。**第一波：大方向谁在犯罪**（forward / backward / renderer /
misc），不做细粒度拆分。

不碰 main.py / brush / loss / coarse_to_fine 源码——import 后替换函数对象（class /
module 属性）。手动跑测（占 shell 久），结果打印到 stdout + 落盘 Debug/out/。

测试参数（覆盖 main 的 module 全局，A 方向运行时改，不改源码）：
  _N_STROKES         = 1    每级 1 批（快速冲到 level 5 grid 16x16，B=256）
  _EPOCHS_PER_STROKE = 100  每批优化步数
  → level 5（grid 16×16, B=256）应明显最慢；重点看它。

桶（互斥，覆盖 epoch 内 + batch 间，见末尾"桶定义"）：
  forward_excl  : reparam + 合成 + loss（epoch 前向，不含渲染）
  renderer      : brush.forward 全部调用（fwd / canvas_update / commit 三处都算）
  backward      : loss.backward（由 step 起点 − loss 终点推得，不 patch Tensor.backward）
  step          : optimizer.step
  canvas_upd_excl : canvas 更新的合成部分（不含渲染）
  commit_excl   : commit 非渲染部分（合成 + params_to_full）
  slice / show  : c2f.slice / plt.show
  other         : wall − 上述之和（zero_grad / init_raw / python 开销）

GPU 计时硬规则：每桶包 ``cuda.synchronize``，否则 CUDA async 让 CPU 计时器提前停、
归因全错。代价：消除流水线重叠 → 计时是"假设串行"量，**归因用，非绝对吞吐**。
Agg 后端（OTBRUSH_HEADLESS=1）使 plt.show≈0；若你想看交互式 show 成本另说。

运行：
    set OTBRUSH_HEADLESS=1 && python timing_probe.py
"""

import os
os.environ.setdefault("OTBRUSH_HEADLESS", "1")      # viewer → Agg，plt.show no-op

import time
from time import perf_counter as perf

import matplotlib
matplotlib.use("Agg", force=True)                  # 双保险，在 import main 前
import matplotlib.pyplot as plt

import torch

# import main 触发其顶层 import（含 viewer，读 OTBRUSH_HEADLESS）；此后改其全局
import main
import core_imports as ci
from loss_space import LossSpace, loss_space
from coarse_to_fine import CoarseToFine
import device as dev

# ========================================================================== #
# 测试参数覆盖（A 方向：改 module 全局，不改源码）
# ========================================================================== #
main._N_STROKES = 1
main._EPOCHS_PER_STROKE = 64
# _R_RAND_LO/HI / _ALPHA_INIT / _OPT_LR 沿用 main 现值，不动。

# ========================================================================== #
# 聚合状态
# ========================================================================== #
timings = {}       # level -> {bucket: sec}
counts = {}        # level -> {bucket: count}
level_wall = {}    # level -> sec
level_meta = {}    # level -> (grid_n, n_tiles, patch_hw)

cur_level = [None]       # 当前 level（slice 调用推进）
ctx = []                 # 渲染上下文栈：'fwd' | 'canvas_update' | 'commit'
last_loss_end = [None]   # loss_space.forward 返回时刻（推 backward 用）
_slice_n = [0]           # slice 调用计数 = level 推进
_lvl_start = [None]      # 当前 level 起始时刻


def _sync():
    if dev.device.type == "cuda":
        torch.cuda.synchronize()


def _add(level, bucket, sec, n=1):
    if level is None:
        level = 0
    d = timings.setdefault(level, {})
    d[bucket] = d.get(bucket, 0.0) + sec
    c = counts.setdefault(level, {})
    c[bucket] = c.get(bucket, 0) + n


# ========================================================================== #
# 保留原函数 + 套壳
# ========================================================================== #
_real_reparam = main.reparam
_real_fwd = main._forward_strokes
_real_commit = main._commit_strokes
_real_brush_fwd = ci.BezierUniformBrush.forward
_real_loss = LossSpace.forward
_real_step = torch.optim.RMSprop.step
_real_p2f = CoarseToFine.params_to_full
_real_slice = CoarseToFine.slice
_real_show = matplotlib.pyplot.show


def _w_reparam(raw):
    _sync(); t0 = perf()
    out = _real_reparam(raw)
    _sync(); _add(cur_level[0], "reparam", perf() - t0)
    return out


def _w_fwd(params, canvas_batch, brush, patch_size):
    # epoch 前向 params 带 grad；canvas 更新 params detached → 靠此区分两路
    is_upd = not params.requires_grad
    c = "canvas_update" if is_upd else "fwd"
    ctx.append(c)
    _sync(); t0 = perf()
    out = _real_fwd(params, canvas_batch, brush, patch_size)
    _sync(); dt = perf() - t0
    ctx.pop()
    _add(cur_level[0], "cu_total" if is_upd else "fwd_total", dt)
    return out


def _w_commit(params_full, canvas_full, brush):
    ctx.append("commit")
    _sync(); t0 = perf()
    out = _real_commit(params_full, canvas_full, brush)
    _sync(); dt = perf() - t0
    ctx.pop()
    _add(cur_level[0], "commit_total", dt)
    return out


def _w_brush_fwd(self, params, patch_size):
    # 渲染上下文由调用方栈顶决定；renderer 归到对应子桶
    c = ctx[-1] if ctx else "other"
    key = {"fwd": "renderer_fwd",
           "canvas_update": "renderer_cu",
           "commit": "renderer_cmt"}.get(c, "renderer_other")
    _sync(); t0 = perf()
    out = _real_brush_fwd(self, params, patch_size)
    _sync(); _add(cur_level[0], key, perf() - t0)
    return out


def _w_loss(self, *a, **kw):
    _sync(); t0 = perf()
    out = _real_loss(self, *a, **kw)
    _sync(); now = perf()
    _add(cur_level[0], "loss", now - t0)
    last_loss_end[0] = now          # 推 backward = step 起点 − 此时刻
    return out


# ---- 第二波：fwd_excl 细拆 = reparam + 合成 + loss{ l1/ot/grad/area } ----
# 四个 loss 原语各自计时（patch 实例方法）。合成 = fwd_total − renderer_fwd − loss
# 由报告端推，不单独 patch。loss 内部四子桶 + visual + 加权组合 = loss 总，可对账。
_real_l1_fwd = loss_space.l1.forward
_real_ot_fwd = loss_space.ot.forward
_real_grad_fwd = loss_space.grad.forward
_real_area_fwd = loss_space.area.forward


def _wrap_loss_prim(name):
    real = {"l1": _real_l1_fwd, "ot": _real_ot_fwd,
            "grad": _real_grad_fwd, "area": _real_area_fwd}[name]
    def w(self, *a, **kw):
        _sync(); t0 = perf()
        out = real(self, *a, **kw)
        _sync(); _add(cur_level[0], f"loss_{name}", perf() - t0)
        return out
    return w

_w_l1 = _wrap_loss_prim("l1")
_w_ot = _wrap_loss_prim("ot")
_w_grad = _wrap_loss_prim("grad")
_w_area = _wrap_loss_prim("area")


# ---- 第三波：OT 内部完备拆分（前向：点云化 + Sinkhorn；反向不探）----
# OT.forward = to_points(pred) + to_points(target) + _loss_fn(Sinkhorn, 含 chunk_b 分块)
# to_points 是同方法两次调（pred 带 grad / target detach），靠 img.requires_grad 区分。
# _loss_fn 是 self._loss_fn（geomloss SamplesLoss），含 chunk_b 串行分块循环。
_real_ot = loss_space.ot
_real_to_points = _real_ot.to_points
_real_sinkhorn = _real_ot._loss_fn
_ot_ctx = []   # 'pred' | 'target'，由 to_points 调用方传入时判定


def _w_to_points(self_real, *a, **kw):
    # 实例属性 patch：self.to_points(pred) → 查实例属性 _w_to_points → _w_to_points(pred)，
    # pred 错位进 self_real，a=()。_real_to_points 是 bound method（绑 loss_space.ot），
    # 调用时需把错位的 pred 作为 img 传回：_real_to_points(self_real) = to_points(img=pred)。
    img = self_real
    tag = "pred" if img.requires_grad else "target"
    _ot_ctx.append(tag)
    _sync(); t0 = perf()
    out = _real_to_points(self_real, *a, **kw)
    _sync(); _add(cur_level[0], f"ot_points_{tag}", perf() - t0)
    _ot_ctx.pop()
    return out


def _w_sinkhorn(*a, **kw):
    # _loss_fn 是 SamplesLoss 对象（非 bound method，不绑 self）。forward 里
    # self._loss_fn(a,x,b,y) → 查实例属性 _w_sinkhorn → _w_sinkhorn(a,x,b,y)（全进 *a）。
    # _real_sinkhorn = SamplesLoss 对象，调用需传全部参数（self=对象已绑在它身上）。
    _sync(); t0 = perf()
    out = _real_sinkhorn(*a, **kw)
    _sync(); _add(cur_level[0], "ot_sinkhorn", perf() - t0)
    return out


def _w_step(self, closure=None):
    # backward = now（flush backward kernel 后）− loss 终点
    _sync(); now = perf()
    le = last_loss_end[0]
    if le is not None and cur_level[0] is not None:
        _add(cur_level[0], "backward", now - le)
    t0 = now
    r = _real_step(self, closure)
    _sync(); _add(cur_level[0], "step", perf() - t0)
    return r


def _w_p2f(self, params, level):
    _sync(); t0 = perf()
    out = _real_p2f(self, params, level)
    _sync(); _add(cur_level[0], "params_to_full", perf() - t0)
    return out


def _w_slice(self, img):
    _sync(); t0 = perf()
    _slice_n[0] += 1
    k = _slice_n[0]
    if _lvl_start[0] is not None:
        level_wall[k - 1] = t0 - _lvl_start[0]      # 上一级 wall 收尾
    cur_level[0] = k
    _lvl_start[0] = t0
    out = _real_slice(self, img)
    _sync(); _add(k, "slice", perf() - t0)
    # out 是 List[Level]；当前级 = out[k-1]，取 grid/B/patch 元信息
    try:
        lo = out[k - 1]
        level_meta[k] = (int(lo.grid_n), int(lo.n_tiles), tuple(int(x) for x in lo.patch_hw))
    except Exception:
        pass
    return out


def _w_show(*a, **kw):
    _sync(); t0 = perf()
    r = _real_show(*a, **kw)
    _sync(); _add(cur_level[0], "show", perf() - t0)
    return r


# ---- 装载（class / module 属性替换）----
main.reparam = _w_reparam
main._forward_strokes = _w_fwd
main._commit_strokes = _w_commit
ci.BezierUniformBrush.forward = _w_brush_fwd
LossSpace.forward = _w_loss
torch.optim.RMSprop.step = _w_step
CoarseToFine.params_to_full = _w_p2f
CoarseToFine.slice = _w_slice
matplotlib.pyplot.show = _w_show
# 第二波：四个 loss 原语实例方法（loss_space 单例上）
loss_space.l1.forward = _w_l1
loss_space.ot.forward = _w_ot
loss_space.grad.forward = _w_grad
loss_space.area.forward = _w_area
# 第三波：OT 内部点云化 + Sinkhorn（patch 到实例属性，同前两波口径）
# 注：实例属性 patch 时调用方传的首参(pred)会进 wrapper 的 self 形参（错位），但
# _real_* 是 bound method（绑了 loss_space.ot），用自己绑的 self，pred 进真实首位参数——
# 与前两波(l1/ot.forward)同机制。故 wrapper 签名必须用 (self, *a, **kw) 吞掉错位 self。
loss_space.ot.to_points = _w_to_points
loss_space.ot._loss_fn = _w_sinkhorn


# ========================================================================== #
# 报告
# ========================================================================== #
def _hms(sec):
    if sec >= 1.0:
        return f"{sec:7.3f}s"
    return f"{sec * 1000:7.2f}ms"


def _pct(x, total):
    if total <= 0:
        return "   -  "
    return f"{100.0 * x / total:5.1f}%"


def _print_report(total_wall):
    print()
    print("=" * 96)
    print("timing_probe 报告 — main 耗时归因（A 方向 monkeypatch；cuda.synchronize 包桶，假设串行量）")
    print("=" * 96)
    print(f"device={dev.device}  config: _N_STROKES={main._N_STROKES}  "
          f"_EPOCHS_PER_STROKE={main._EPOCHS_PER_STROKE}  "
          f"levels={_slice_n[0]}  total_wall={_hms(total_wall)}")
    print("-" * 96)

    levels = sorted(timings.keys() | set(level_wall.keys()))
    levels = [l for l in levels if l > 0]
    if not levels:
        print("（无 level 数据；main 可能未进入循环即崩溃）")
        return

    # ---- Table A：每级阶段耗时（互斥，% = 占该级 wall）----
    print()
    print("Table A — 每级阶段耗时（互斥；% 占该级 wall）")
    print("-" * 96)
    hdr = (f"{'Lvl':>3} {'grid':>5} {'B':>4} {'patch':>9} {'wall':>9}  "
           f"{'fwd_excl':>9} {'renderer':>9} {'backward':>9} {'step':>9}  "
           f"{'cmt_excl':>9} {'cu_excl':>9} {'slice':>8} {'show':>8} {'other':>8}")
    print(hdr)
    print("-" * 96)
    sum_wall = 0.0
    sum_bucket = {k: 0.0 for k in
                  ["fwd_excl", "renderer", "backward", "step",
                   "cmt_excl", "cu_excl", "slice", "show", "other"]}
    for k in levels:
        b = timings.get(k, {})
        cnt = counts.get(k, {})
        grid, B, patch = level_meta.get(k, ("?", "?", "?"))
        patch_s = f"{patch[0]}×{patch[1]}" if isinstance(patch, tuple) else str(patch)

        reparam_t = b.get("reparam", 0.0)
        fwd_total = b.get("fwd_total", 0.0)
        cu_total = b.get("cu_total", 0.0)
        commit_total = b.get("commit_total", 0.0)
        loss_t = b.get("loss", 0.0)
        step_t = b.get("step", 0.0)
        bw_t = b.get("backward", 0.0)
        r_fwd = b.get("renderer_fwd", 0.0)
        r_cu = b.get("renderer_cu", 0.0)
        r_cmt = b.get("renderer_cmt", 0.0)
        p2f_t = b.get("params_to_full", 0.0)
        slice_t = b.get("slice", 0.0)
        show_t = b.get("show", 0.0)

        composite_fwd = fwd_total - r_fwd
        composite_cu = cu_total - r_cu
        composite_cmt = commit_total - r_cmt

        fwd_excl = reparam_t + composite_fwd + loss_t
        renderer = r_fwd + r_cu + r_cmt
        cu_excl = composite_cu
        cmt_excl = composite_cmt + p2f_t

        wall = level_wall.get(k, 0.0)
        attrib = (fwd_excl + renderer + bw_t + step_t +
                  cmt_excl + cu_excl + slice_t + show_t)
        other = max(0.0, wall - attrib)

        sum_wall += wall
        sum_bucket["fwd_excl"] += fwd_excl
        sum_bucket["renderer"] += renderer
        sum_bucket["backward"] += bw_t
        sum_bucket["step"] += step_t
        sum_bucket["cmt_excl"] += cmt_excl
        sum_bucket["cu_excl"] += cu_excl
        sum_bucket["slice"] += slice_t
        sum_bucket["show"] += show_t
        sum_bucket["other"] += other

        print(f"{k:>3} {str(grid)+'x'+str(grid):>5} {B:>4} {patch_s:>9} {_hms(wall):>9}  "
              f"{_hms(fwd_excl):>9} {_hms(renderer):>9} {_hms(bw_t):>9} {_hms(step_t):>9}  "
              f"{_hms(cmt_excl):>9} {_hms(cu_excl):>9} {_hms(slice_t):>8} {_hms(show_t):>8} {_hms(other):>8}")
    print("-" * 96)
    print(f"{'SUM':>3} {'':>5} {'':>4} {'':>9} {_hms(sum_wall):>9}  "
          f"{_hms(sum_bucket['fwd_excl']):>9} {_hms(sum_bucket['renderer']):>9} "
          f"{_hms(sum_bucket['backward']):>9} {_hms(sum_bucket['step']):>9}  "
          f"{_hms(sum_bucket['cmt_excl']):>9} {_hms(sum_bucket['cu_excl']):>9} "
          f"{_hms(sum_bucket['slice']):>8} {_hms(sum_bucket['show']):>8} {_hms(sum_bucket['other']):>8}")
    setup_resid = max(0.0, total_wall - sum_wall)
    print(f"\n  setup（img_input.load / c2f 建 pyramid，首 slice 之前）= {_hms(setup_resid)}")

    # ---- Table B：renderer 上下文拆分 + 计数 + per-call ----
    print()
    print("Table B — renderer 上下文拆分（哪个渲染最贵；含调用数与 per-call 均值）")
    print("-" * 96)
    hdr2 = (f"{'Lvl':>3}  {'renderer_fwd':>14} {'renderer_cu':>14} {'renderer_cmt':>16}  "
            f"{'renderer_total':>14} {'%wall':>6}")
    print(hdr2)
    print("-" * 96)
    for k in levels:
        b = timings.get(k, {})
        cnt = counts.get(k, {})
        wall = level_wall.get(k, 0.0)

        def cell(key, t):
            n = cnt.get(key, 0)
            t = b.get(key, 0.0)
            avg = (t / n * 1000) if n else 0.0
            return f"{_hms(t)}×{n}({avg:.1f}ms)"

        rf = cell("renderer_fwd", 0)
        rcu = cell("renderer_cu", 0)
        rcm = cell("renderer_cmt", 0)
        tot = b.get("renderer_fwd", 0) + b.get("renderer_cu", 0) + b.get("renderer_cmt", 0)
        print(f"{k:>3}  {rf:>14} {rcu:>14} {rcm:>16}  {_hms(tot):>14} {_pct(tot, wall):>6}")
    print("-" * 96)
    print("  注：renderer_cmt 在 level 5 应为 256 次全图(1024²)串行渲染——若此列最大，")
    print("     则 commit 全图串行光栅化是罪犯（与 epoch forward 的 patch-res 渲染不同量级）。")

    # ---- Table C：每 epoch 均值（看随 B 的扩展性）----
    print()
    print("Table C — 每 epoch 均值（loss 调用数 = epochs；看随 B 的单步扩展性）")
    print("-" * 96)
    hdr3 = (f"{'Lvl':>3} {'B':>4} {'epochs':>7}  "
            f"{'fwd_excl/ep':>12} {'rdr_fwd/ep':>12} {'backward/ep':>12} {'step/ep':>10} {'wall/ep':>10}")
    print(hdr3)
    print("-" * 96)
    for k in levels:
        b = timings.get(k, {})
        cnt = counts.get(k, {})
        grid, B, patch = level_meta.get(k, ("?", "?", (0, 0)))
        ep = cnt.get("loss", 0)
        if not ep:
            continue
        reparam_t = b.get("reparam", 0.0)
        composite_fwd = b.get("fwd_total", 0.0) - b.get("renderer_fwd", 0.0)
        fwd_excl = reparam_t + composite_fwd + b.get("loss", 0.0)
        # reparam 多算了固化那次（+batches），按 epoch 摊减可忽略
        rf = b.get("renderer_fwd", 0.0)
        bw = b.get("backward", 0.0)
        st = b.get("step", 0.0)
        wall = level_wall.get(k, 0.0)
        print(f"{k:>3} {B:>4} {ep:>7}  "
              f"{_hms(fwd_excl/ep):>12} {_hms(rf/ep):>12} {_hms(bw/ep):>12} "
              f"{_hms(st/ep):>10} {_hms(wall/ep):>10}")

    # ---- Table D：fwd_excl 细拆 = reparam + 合成 + loss{l1/ot/grad/area} ----
    # 第二波：定位 fwd_excl 里真凶。合成 = fwd_total − renderer_fwd − loss（推得）。
    # loss 总 vs 四子桶之和的差 = visual + 加权组合 + set_target 开销（应为小）。
    print()
    print("Table D — fwd_excl 细拆（reparam / 合成 / loss 四子桶；% 占 fwd_excl）")
    print("-" * 96)
    hdr4 = (f"{'Lvl':>3} {'B':>4}  {'fwd_excl':>10}  {'reparam':>9} {'composite':>10} "
            f"{'loss_total':>10}  {'l1':>9} {'ot':>9} {'grad':>9} {'area':>9}  "
            f"{'loss_diff':>9}")
    print(hdr4)
    print("-" * 96)
    sum_d = {k: 0.0 for k in ["fwd_excl", "reparam", "composite", "loss_total",
                              "l1", "ot", "grad", "area", "loss_diff"]}
    for k in levels:
        b = timings.get(k, {})
        cnt = counts.get(k, {})
        grid, B, patch = level_meta.get(k, ("?", "?", (0, 0)))
        reparam_t = b.get("reparam", 0.0)
        loss_t = b.get("loss", 0.0)
        # composite = _forward_strokes 总 − 其中 brush.forward；loss 不在 _forward_strokes 内
        composite = b.get("fwd_total", 0.0) - b.get("renderer_fwd", 0.0)
        fwd_excl = reparam_t + composite + loss_t
        l1 = b.get("loss_l1", 0.0)
        ot = b.get("loss_ot", 0.0)
        grad = b.get("loss_grad", 0.0)
        area = b.get("loss_area", 0.0)
        prim_sum = l1 + ot + grad + area
        loss_diff = loss_t - prim_sum     # visual + 加权组合 + set_target
        for kk, vv in [("fwd_excl", fwd_excl), ("reparam", reparam_t),
                       ("composite", composite), ("loss_total", loss_t),
                       ("l1", l1), ("ot", ot), ("grad", grad),
                       ("area", area), ("loss_diff", loss_diff)]:
            sum_d[kk] += vv
        print(f"{k:>3} {B:>4}  {_hms(fwd_excl):>10}  "
              f"{_hms(reparam_t):>9} {_hms(composite):>10} {_hms(loss_t):>10}  "
              f"{_hms(l1):>9} {_hms(ot):>9} {_hms(grad):>9} {_hms(area):>9}  "
              f"{_hms(loss_diff):>9}")
    print("-" * 96)
    print(f"{'SUM':>3} {'':>4}  {_hms(sum_d['fwd_excl']):>10}  "
          f"{_hms(sum_d['reparam']):>9} {_hms(sum_d['composite']):>10} {_hms(sum_d['loss_total']):>10}  "
          f"{_hms(sum_d['l1']):>9} {_hms(sum_d['ot']):>9} {_hms(sum_d['grad']):>9} {_hms(sum_d['area']):>9}  "
          f"{_hms(sum_d['loss_diff']):>9}")
    print("-" * 96)
    print("  composite = fwd_total − renderer_fwd（推得，非直接 patch；loss 不在 _forward_strokes 内）")
    print("  loss_diff = loss_total − (l1+ot+grad+area) = visual + 加权组合 + set_target（应小）")
    print("  看：哪个 loss 子桶随 B 爆炸 → 即 fwd_excl 真凶；reparam/composite 应稳定小。")

    # ---- Table E：loss 四子桶 per-epoch + per-call（看随 B 的扩展性）----
    print()
    print("Table E — loss 四子桶每 epoch 均值 + per-call（看随 B 扩展性）")
    print("-" * 96)
    hdr5 = (f"{'Lvl':>3} {'B':>4}  {'l1/ep':>9} {'ot/ep':>9} {'grad/ep':>9} {'area/ep':>9}  "
            f"{'l1/call':>9} {'ot/call':>9} {'grad/call':>11} {'area/call':>11}")
    print(hdr5)
    print("-" * 96)
    for k in levels:
        b = timings.get(k, {})
        cnt = counts.get(k, {})
        grid, B, patch = level_meta.get(k, ("?", "?", (0, 0)))
        ep = cnt.get("loss", 0)
        if not ep:
            continue

        def cell(key):
            n = cnt.get(key, 0)
            t = b.get(key, 0.0)
            per_ep = t / ep if ep else 0.0
            per_call = (t / n * 1000) if n else 0.0
            return per_ep, per_call, n

        l1_ep, l1_pc, l1_n = cell("loss_l1")
        ot_ep, ot_pc, ot_n = cell("loss_ot")
        grad_ep, grad_pc, grad_n = cell("loss_grad")
        area_ep, area_pc, area_n = cell("loss_area")
        print(f"{k:>3} {B:>4}  "
              f"{_hms(l1_ep):>9} {_hms(ot_ep):>9} {_hms(grad_ep):>9} {_hms(area_ep):>9}  "
              f"{f'{l1_pc:.2f}ms×{l1_n}':>9} {f'{ot_pc:.2f}ms×{ot_n}':>9} "
              f"{f'{grad_pc:.2f}ms×{grad_n}':>11} {f'{area_pc:.2f}ms×{area_n}':>11}")
    print("-" * 96)
    print("  ot/call 随 B 应明显涨（点云化 + Sinkhorn 随 B 扩展）；l1/grad 随像素数，B 涨它涨。")

    # ---- Table F：OT 内部完备拆分（点云化 pred/target + Sinkhorn；反向不探）----
    # 第三波：OT.forward = to_points(pred) + to_points(target) + _loss_fn(Sinkhorn)。
    # to_points 同方法两次调（pred 带 grad / target detach，靠 requires_grad 区分）。
    # _loss_fn 含 chunk_b 分块：B=256,chunk_b=16 → 16 次串行调用本函数。
    # 对账：ot_points_pred + ot_points_target + ot_sinkhorn ≈ loss_ot（差=info 构建+归一化尾巴）。
    print()
    print("Table F — OT 内部完备拆分（点云化 pred/target + Sinkhorn；仅前向，反向不探）")
    print("-" * 96)
    hdr6 = (f"{'Lvl':>3} {'B':>4}  {'ot_total':>10}  {'pts_pred':>10} {'pts_target':>11} "
            f"{'sinkhorn':>10}  {'pts+sk/ot':>10}  {'sk_calls':>8} {'sk/call':>9}")
    print(hdr6)
    print("-" * 96)
    sum_f = {k: 0.0 for k in ["ot_total", "pts_pred", "pts_target", "sinkhorn"]}
    sum_sk_calls = 0
    for k in levels:
        b = timings.get(k, {})
        cnt = counts.get(k, {})
        grid, B, patch = level_meta.get(k, ("?", "?", (0, 0)))
        ot_total = b.get("loss_ot", 0.0)
        pts_pred = b.get("ot_points_pred", 0.0)
        pts_tgt = b.get("ot_points_target", 0.0)
        sinkhorn = b.get("ot_sinkhorn", 0.0)
        sk_calls = cnt.get("ot_sinkhorn", 0)
        sk_per = (sinkhorn / sk_calls * 1000) if sk_calls else 0.0
        pts_sk = pts_pred + pts_tgt + sinkhorn
        ratio = (pts_sk / ot_total) if ot_total > 0 else 0.0
        for kk, vv in [("ot_total", ot_total), ("pts_pred", pts_pred),
                       ("pts_target", pts_tgt), ("sinkhorn", sinkhorn)]:
            sum_f[kk] += vv
        sum_sk_calls += sk_calls
        print(f"{k:>3} {B:>4}  {_hms(ot_total):>10}  "
              f"{_hms(pts_pred):>10} {_hms(pts_tgt):>11} {_hms(sinkhorn):>10}  "
              f"{f'{ratio*100:.1f}%':>10}  {sk_calls:>8} {f'{sk_per:.2f}ms':>9}")
    print("-" * 96)
    print(f"{'SUM':>3} {'':>4}  {_hms(sum_f['ot_total']):>10}  "
          f"{_hms(sum_f['pts_pred']):>10} {_hms(sum_f['pts_target']):>11} "
          f"{_hms(sum_f['sinkhorn']):>10}  {'':>10}  {sum_sk_calls:>8}")
    print("-" * 96)
    print("  pts+sk/ot = (pts_pred+pts_target+sinkhorn) / ot_total；应≈100%（差=info构建+归一尾巴）")
    print("  sk_calls  = _loss_fn 调用数 = ceil(B/chunk_b)；B=256,chunk_b=16 → 16")
    print("  sk/call   = sinkhorn / sk_calls；看单块 Sinkhorn 随 B_chunk 扩展性")
    print("  注：OT 反向（loss.backward 穿 OT 段）隐含在 backward 总桶，未单独拆——")
    print("     autograd 图穿 OT→渲染→params，强行 autograd.grad 拆会改图重复计算，不探。")

    print()
    print("=" * 96)
    print("桶定义（互斥）：")
    print("  fwd_excl   = reparam + 合成(cat/over) + loss   [epoch 前向，不含渲染]")
    print("  renderer   = brush.forward 全部（fwd + canvas_update + commit 三处）")
    print("  backward   = loss.backward（step 起点 − loss 终点推得）")
    print("  step       = optimizer.step")
    print("  cmt_excl   = commit 非渲染（合成 + params_to_full）")
    print("  cu_excl    = canvas 更新非渲染（合成）")
    print("  other      = wall − 上述（zero_grad / init_raw / python 开销）")
    print("第二波（Table D/E）：fwd_excl 细拆 = reparam + 合成 + loss{l1/ot/grad/area}")
    print("  四个 loss 原语实例方法单独计时；合成 = fwd_total − renderer_fwd（推得）")
    print("第三波（Table F）：OT 内部 = to_points(pred/target) + sinkhorn（_loss_fn 含分块）")
    print("  to_points 靠 img.requires_grad 区分 pred/target；sinkhorn 调用数 = ceil(B/chunk_b)")
    print("=" * 96)


# ========================================================================== #
# 入口
# ========================================================================== #
def _save_log(text):
    try:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Debug", "out")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "timing_report.txt"), "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"[probe] 落盘失败：{e}")


def run():
    print("=" * 96)
    print("timing_probe — main 耗时归因（A 方向 monkeypatch；不改源码）")
    print("=" * 96)
    print(f"device={dev.device}  config: _N_STROKES={main._N_STROKES}  "
          f"_EPOCHS_PER_STROKE={main._EPOCHS_PER_STROKE}")
    print("patches: reparam / _forward_strokes / _commit_strokes / brush.forward /")
    print("         loss_space.forward / RMSprop.step / params_to_full / c2f.slice / plt.show")
    print("（cuda.synchronize 包每个桶；计时为假设串行量，归因用）")
    print("-" * 96)

    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()

    t0 = perf()
    err = None
    try:
        with redirect_stdout(buf):
            main.main()
    except Exception as e:
        err = e
    _sync()
    total_wall = perf() - t0
    # 末级 wall 收尾：slice 仅在"下一级开头"收尾上一级；末级无下一级，main 返回后在此补。
    # 不补则 level_wall[末级]=0 → wall/other/%wall/wall-per-ep 全失真，末级真实 wall
    # 被错记进 setup（上一轮 setup 虚高 103s 即此因）。
    if _lvl_start[0] is not None and _slice_n[0] > 0:
        level_wall[_slice_n[0]] = (t0 + total_wall) - _lvl_start[0]

    # main 的正常打印先回放
    print(buf.getvalue())

    # 捕获报告到字符串（同时打印 + 落盘）
    rep_buf = io.StringIO()
    import sys
    _orig = sys.stdout
    class _Tee:
        def __init__(self, *s): self.s = s
        def write(self, x):
            for ss in self.s: ss.write(x)
        def flush(self):
            for ss in self.s:
                try: ss.flush()
                except Exception: pass
    sys.stdout = _Tee(_orig, rep_buf)
    try:
        _print_report(total_wall)
        if err is not None:
            print()
            print(f"[probe] main 抛异常（已捕，部分数据可能缺失）：{type(err).__name__}: {err}")
    finally:
        sys.stdout = _orig

    _save_log(buf.getvalue() + rep_buf.getvalue())
    if err is not None:
        raise err


if __name__ == "__main__":
    run()
