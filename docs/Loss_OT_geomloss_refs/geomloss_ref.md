# geomloss 参考文档（0.3.1 实测 + 源码核对）

> 本文件替代已删除的 `tensorized_bottleneck.md`。旧文档基于 Win + 旧环境的 timing 结论，
> 路径引用还是 `loss-calculate-workspace/loss_calculate_core/implementations/...`，已过时。
> 本文档以 **geomloss 0.3.1 现场实测 + site-packages 源码核对** 为主，官方文档为辅，
> 记录可用 API、安装前置、backend/keops 关系，以及对 `losscal` 重构的决定性影响。
>
> **关键事实**：geomloss **0.3.1 即 PyPI 最新版**（0.3.0 已 yanked；可用 0.2~0.3.1）。
> 不存在比 0.3.1 更新的版本。官方文档（首页 + pytorch-api.html）与 0.3.1 代码**存在不一致**：
> 首页广告的 `ImagesLoss`/`VolumesLoss` 在 0.3.1 代码里未实现；API 页未文档 `sinkhorn_divergence`/
> `ImagesBarycenter`/`ot.solve_grid` 但代码里都有。**以实测 + 源码为准**。
>
> 信息出处标注：
> - `[官网]` = kernel-operations.io 抓取页
> - `[官网API]` = 官方 pytorch-api.html 页（不完整、与代码不一致；本目录原快照 `geomloss_pytorch_api.txt` 已删除）
> - `[实测]` = 本机 geomloss 0.3.1 + torch 2.12.1+cu130 实跑结果
> - `[源码]` = `.venv/lib/python3.14/site-packages/geomloss/` 源码浏览（最权威）

---

## 1. 安装

### 1.1 geomloss 本体 `[官网: install.html]`
```bash
pip install geomloss            # 推荐；不含 pykeops，backend 只能 tensorized
pip install geomloss[full]      # 带可选依赖（含 pykeops），Colab 用
```
- 前置仅需 PyTorch。纯 Python，**不需要 nvcc / CUDA toolkit**。
- 本机已装：**geomloss 0.3.1**（顺带 scipy 1.18.0）。`[实测]`

### 1.2 pykeops（online/multiscale backend 的可选前置）`[官网: keops/python/installation.html]`
```bash
pip install pykeops
```
硬性要求：
- Python ≥ 3.8 + numpy
- **C++ 编译器**：g++ ≥ 7 或 clang++ ≥ 8（C++11）
- **系统级 CUDA toolkit ≥ 10.0**（含 `nvcc` + `cuda.h`）—— 注意：conda 里装的 CUDA toolkit **不够**，
  编译时找不到 `cuda.h`，必须系统级安装
- PyTorch ≥ 1.5（可选）

编译产物缓存 `~/.cache/keops/`（`KEOPS_CACHE_FOLDER` 可改位置）。自测：
```python
import pykeops
pykeops.test_torch_bindings()   # 或 pykeops.check_health()
pykeops.clean_pykeops()         # 出问题先清缓存
```

---

## 2. 本机环境现状（已装齐）

> 里程碑 1 已完成环境打通：geomloss + pykeops + cuda-toolkit 全部就位，pykeops 自检通过。

| 组件 | 状态 | 说明 |
|---|---|---|
| 平台 | ✅ WSL2 Ubuntu 24.04.4 | 内核 `...-microsoft-standard-WSL2`；GPU 由 Windows 驱动透传 |
| gcc / g++ | ✅ 13.3.0 | 远超 keops 要求（≥7, C++11）；在 CUDA 13.3 支持矩阵内（6.x–15.x） |
| Windows 驱动 / runtime CUDA | ✅ 581.57 / 13.0 | `nvidia-smi` 有，`torch.cuda.is_available()=True`；WSL2 下 nvidia-smi 是 limited feature set |
| **系统级 CUDA toolkit** | ✅ **已装 cuda-toolkit-13-3** | `nvcc` release 13.3 V13.3.73、`/usr/local/cuda/include/cuda.h` 在；WSL-Ubuntu 专用包，不含驱动 |
| Python / numpy | ⚠️ 3.14.6 / 2.4.4 | 极新；pykeops 2.3 发布早于 py3.14，但源码编译实测通过 |
| torch | ✅ 2.12.1+cu130 | CUDA 13.0 |
| **geomloss** | ✅ 0.3.1 | `[实测]` |
| **pykeops** | ✅ 2.3 | `[实测]` `test_torch_bindings()` 通过，KeOps JIT 引擎 + Sum((a-b)**2) reduction kernel 在 GPU 跑通 |

### 2.1 WSL2 + CUDA 13.3 的 arch 坑（排错记录）
- toolkit 13.3 比 WSL2 透传的驱动 13.0 新 0.3 个 minor。nvcc 13.3 **默认**生成的 PTX 被 driver 13.0 拒：
  `the provided PTX was compiled with an unsupported toolchain`。
- **显式 `-arch=sm_86` 生成 SASS 即绕过**（本机 GPU = RTX 3060, compute capability 8.6）。
- **pykeops 不受影响**：它自动检测 GPU compute capability 并传 `-arch=sm_XX` 给 nvcc，自检一次通过。
- WSL2 仓库（`file:/var/cuda-repo-wsl-ubuntu-13-3-local`）只有 13.3，无 13-0 可降级；保持 13.3 即可。

### 2.2 WSL2 安装 toolkit 的关键约束 `[官网: wsl-user-guide]`
- 必须用 **WSL-Ubuntu** 专用 installer（下载页 Distribution 选 `WSL-Ubuntu`，不是 `Ubuntu`），
  或退而用普通 Ubuntu 仓库但**只装 `cuda-toolkit`**，**绝不能装 `cuda`/`cuda-drivers`**——
  会覆盖 WSL2 透传的 `libcuda.so` stub、搞坏 GPU。
- WSL2 下 pinned memory 受限；nvidia-smi 在 `/usr/lib/wsl/lib/`，且是 limited feature set（不影响 pykeops）。

---

## 3. API 清单（0.3.1 实测 + 源码核对为准）

> 全部以 Python `inspect` + 真跑实测 + site-packages 源码浏览为准。官方 API 页 `[官网API]` 的错漏见 3.7。

### 3.0 包结构 `[源码]`

```
geomloss/
├─ __init__.py(9)          顶层导出 SamplesLoss / ImagesBarycenter / sinkhorn_divergence
├─ _arguments.py(154)      check_regularization / check_marginal / ArrayProperties
├─ _cache.py(90)           cached property 机制（cache_methods_and_properties）
├─ _typing.py(159)         RealTensor / CostMatrices / 类型别名
├─ _backends/              keops/torch/numpy 三后端抽象（bk.keops_available / bk.LazyTensor）
├─ _input_validation/      convert_inputs 装饰器
├─ _legacy/                SamplesLoss + sinkhorn_divergence + sinkhorn_images/samples + barycenter_images
└─ ot/
   ├─ __init__.py(26)      导出 solve/solve_batch/solve_sample/solve_sample_batch/solve_grid/barycenter*/OTResult*
   ├─ _ot_result.py(454)   OTResult 抽象基类 + LinearOperator
   ├─ _abstract_solvers/   sinkhorn_loop / annealing_parameters / sinkhorn_cost / unbalanced_ot / barycenters
   └─ _implementations/    matrix.py / sample.py / grid.py
```

- **两套 OT 体系并存**：`_legacy/`（SamplesLoss 等，成熟、功能全、支持 multiscale）与 `ot/`（新 API，
  solve/solve_sample 等，部分坏桩）。`geomloss.__init__` 从 `_legacy` re-export 顶层三接口。
- `_backends` 抽象让 sample/matrix 实现同时支持 numpy/torch/keops 三种数组库；`bk.keops_available`
  控制 cost matrix 走 lazy（KeOps LazyTensor）还是 dense。

### 3.1 顶层接口 `[实测: dir + inspect + 源码: __init__.py]`

去掉 `os`/`sys`（标准库），0.3.1 顶层公开 3 个 geomloss 接口：

| 接口 | 类型 | 签名 | 实测可用 | 定位 |
|---|---|---|---|---|
| `SamplesLoss` | class(nn.Module) | `(loss='sinkhorn', p=2, blur=0.05, reach=None, diameter=None, scaling=0.5, truncate=5, cost=None, kernel=None, cluster_scale=None, debias=True, potentials=False, verbose=False, backend='auto')` | ✅ 点云 | 主力 criterion，见 3.2 |
| `sinkhorn_divergence` | function | `(a, b, p=2, blur=None, reach=None, axes=None, scaling=0.5, cost=None, debias=True, potentials=False, verbose=False, **kwargs)` | ❌ grid 坏 | 实际定义于 `geomloss._legacy.sinkhorn_divergence`，legacy grid 路径，见 3.5 |
| `ImagesBarycenter` | function | `(measures, weights, blur=0, p=2, scaling_N=10, backward_iterations=5)` | ❌ 坏 | legacy 图像 barycenter，见 3.5 |

- 🆕 `[源码]` **`__all__` 是 bug**：`__all__ = sorted(["SamplesLoss, ImagesBarycenter"])` 是**单元素列表**
  （字符串内含逗号，非两元素），`from geomloss import *` 实际啥也不导出。显式 `from geomloss import X` 不受影响。
- **`ImagesLoss` / `VolumesLoss` 在 0.3.1 不存在** `[实测: hasattr=False]`——`[官网首页]` 广告了这两个类，
  但 0.3.1（最新版）代码未实现，属文档超前于代码。
- `sinkhorn_divergence` / `ImagesBarycenter` 在 0.3.1 代码里存在（定义于 `geomloss._legacy` 子包），
  但 `[官网API]` 未文档——属文档未覆盖代码已有接口。

### 3.2 `geomloss.SamplesLoss`（点云 criterion）`[实测 + 源码: _legacy/samples_loss.py]`

`loss=` 五种：
| 值 | 是什么 |
|---|---|
| `"sinkhorn"` | 去偏 Sinkhorn 散度（Wasserstein↔kernel 插值） |
| `"hausdorff"` | 加权 Hausdorff（ICP↔kernel） |
| `"energy"` | Energy Distance MMD，k=-‖x-y‖ |
| `"gaussian"` | 高斯 MMD，k=exp(-‖x-y‖²/2σ²)，σ=blur |
| `"laplacian"` | 拉普拉斯 MMD，k=exp(-‖x-y‖/σ)，σ=blur |

调用约定（点云）：`loss_fn(a, x, b, y)`，a/b 权重 (N,)/(M,)，x/y 坐标 (N,D)/(M,D)。
也支持 2 参数 `(x,y)`（自动均匀权重）或 6 参数 `(l_x,a,x,l_y,b,y)`（带 cluster 标签，仅 multiscale）。
- `reach=None` → balanced OT（要求两边总质量相等）；非 None → unbalanced。
- `debias=True`（**默认**）→ 去偏 Sinkhorn 散度（正定，α=β⇔0）。注意与 `ot.solve_sample` 默认 `debias=False` 相反。
- `potentials=True` → 返回双对偶势 (f,g) 而非标量。`[实测]` ✅ 可跑，f/g shape (1,N)。
- `[源码]` **backend="auto" 选择逻辑**：`M*N <= 5000²` → tensorized；否则 D≤3 且 sinkhorn 且
  `M*N > 10000²` 且 p==2 → multiscale；否则 online。multiscale 不支持 batch>1（会 warn 降级 tensorized）。
- `[源码]` 默认 `truncate=5`（multiscale kernel 截断倍数）、`scaling=0.5`（ε-scaling 下降比）。
- `[实测]` **纯点云接口，不接受图像 grid**：传 `(1,1,H,W)` 报 `ValueError: Input samples 'x' and 'y'
  should be encoded as (N,D) or (B,N,D)`。图像须经点云化才能喂入。

### 3.3 `geomloss.ot` 求解器 `[实测 + 源码: _implementations/{matrix,sample,grid}.py]`

| 函数 | 签名要点 | 输入 | 返回 | 0.3.1 实测 |
|---|---|---|---|---|
| `solve(C, *, reg, a, b, unbalanced, unbalanced_type, method, max_iter, tol)` | 显式代价矩阵 | (N,M) | `OTResultMatrix` | ✅ **需显式 `max_iter`**（默认 None 触发 ValueError） |
| `solve_batch(C, *, reg, a, b, ..., max_iter)` | 批代价矩阵 | (B,N,M) | `OTResultMatrix` | ✅ `.value` 返回 (B,) 向量、`.plan` 返回 (B,N,M) |
| `solve_sample(X_a, X_b, a, b, cost, debias, reg, unbalanced, ..., max_iter, tol, blur, reach)` | 点云坐标 | (N,D)/(M,D) | `OTResultSample` | ✅ **需显式 `max_iter`** |
| `solve_sample_batch(X_a, X_b, a, b, ..., reg=0, ...)` | 批点云 | (B,N,D) | `OTResultSample` | ❌ `NotImplementedError: This function is not implemented yet.`（第一行 raise，后接死代码） |
| `solve_grid(a, b, cost, axes, periodic, p, blur, reach)` | grid 权重 | (B,Nx,Ny) | `OTResult`? | ❌ `NameError: name 'OTResult' is not defined`（坏桩） |

- `[源码]` **`blur`/`reach` 是 `reg`/`unbalanced` 的几何化冗余参数**：`reg = p * blur**p`，
  `unbalanced = p * reach**p`。`solve_sample` 支持用 `blur`（直观的"模糊半径" σ）代替 `reg`，二者互斥。
- `[源码]` `solve_sample` 的 `cost` 只支持 `"sqeuclidean"`（其它 NotImplementedError），p 由 cost 推（sqeuclidean→p=2）。
- `[源码]` `solve_sample` 的 cost matrix 支持 **lazy（keops）和 dense 两种**：`matrix_type="auto"` →
  `bk.keops_available` 为真则 lazy（KeOps LazyTensor）否则 dense。**本机装了 pykeops 后自动走 lazy 高效路径**。
- `[源码]` `solve` 内部调 `solve_batch`（加 dummy batch 维）再 `_squeeze_batchdim()` 去掉。
- 注：`solve_sample_batch` 签名 `reg=0`（与 `solve_sample` 的 `reg=None` 不一致）；`solve_grid` 无
  `reg`/`max_iter`/`debias` 参数，形态明显是半成品。

### 3.4 `geomloss.ot` 结果类与属性 `[实测 + 源码: _ot_result.py + _implementations/*.py]`

| 类 | 用途 | 由谁返回 |
|---|---|---|
| `OTResult` | 抽象基类 | — |
| `OTResultMatrix` | 矩阵 OT 结果 | `solve` / `solve_batch` |
| `OTResultSample` | 点云 OT 结果 | `solve_sample`（`solve_sample_batch` 未实现） |
| `LinearOperator` | 运输计划线性算子：`T`/`shape`(property)、`from_dense`/`from_lazy_tensor`/`rescale`/`transpose` | `*.plan_operator` / `density_operator` |

`[源码]` **property 机制**：`OTResult._cached_properties` 元组列出 15 个名字
（potential_a/b/aa/bb, density/lazy_density/density_operator, plan/lazy_plan/plan_operator,
value, marginal_a/b, a_to_b/b_to_a, citation），由 `cache_methods_and_properties` 把 `_xxx()`
方法动态绑成带缓存的 `xxx` property。子类可覆盖 `_cached_properties` 控制暴露面：
- `OTResultMatrix` 覆盖为去掉 `potential_aa`/`potential_bb`/`a_to_b`/`b_to_a`（matrix 硬编码 debias=False）。
- `OTResultSample` 用基类列表（debias=True 时 `potential_aa`/`potential_bb` 可用）。

`OTResult*` 属性实测（`OTResultSample` 经 `solve_sample(debias=True)`，`OTResultMatrix` 经 `solve`）：

| property | OTResultSample | OTResultMatrix | 说明 |
|---|---|---|---|
| `value` | ✅ 标量 | ✅ 标量 | 损失值；要求 `reg_type=="KL"` 且 `unbalanced_type=="KL"` |
| `plan` | ✅ (N,M) | ✅ (N,M) | 运输计划 |
| `potential_a` / `potential_b` | ✅ (N,) | ✅ (N,) | 双对偶势 f/g |
| `potential_aa` / `potential_bb` | ✅ (N,) | ❌ ValueError（未定义） | 自相互作用势，仅 debiased 点云可用 |
| `marginal_a` / `marginal_b` | ✅ | ✅ | 边际 |
| `density` | ✅ (N,M) | ✅ (N,M) | 计划密度 |
| `density_operator` / `plan_operator` | ✅ LinearOperator | ✅ LinearOperator | `[源码]` 实现完整可用（非坏桩） |
| `lazy_plan` / `lazy_density` | 需 pykeops（已装） | 需 pykeops | KeOps LazyTensor；`OTResultSample` 在 lazy cost 下非 None |
| `a_to_b` / `b_to_a` | None（未实现） | None | 位移向量；`[源码]` 基类 `return None`，两子类均未重写 |
| `citation` | ✅ str | ✅ str | |
| `cache_clear` / `cast` | 方法 | 方法 | |

- `[源码]` **`OTResultSample` 只支持非 batch（B=0）**：`_shapes` 里 `if ap.B == 0: ... else: raise NotImplementedError()`。
- `[源码]` **`LinearOperator` 实现完整**：`from_dense`/`from_lazy_tensor`/`rescale`/`transpose`/`__matmul__`
  全实现，`plan_operator = density_operator.rescale(input_scaling=b, output_scaling=a)`。可支撑将来 losscal
  做 transport 算子/边际 viz。

### 3.5 barycenter 与 legacy grid 接口（0.3.1 全坏）`[实测 + 源码]`

| 接口 | 签名 | 实测 | 源码坏因 |
|---|---|---|---|
| `ot.barycenter(cost, a, weights)` | 矩阵版 | ❌ `NameError: name 'potentials' is not defined` | 函数体 `return OTResult(potentials=potentials, masses=masses)`，`potentials`/`masses` 未定义 |
| `ot.barycenter_sample(xa, a, weights)` | 点云版 | ❌ `NameError: name 'potentials' is not defined` | 同上 |
| `ot.barycenter_grid(a)` | grid 版 | ❌ `NameError: name 'OTResult' is not defined` | `grid.py` 未 `import OTResult`，函数体 `return OTResult(potentials)` |
| `geomloss.sinkhorn_divergence(a, b, ...)` | legacy grid，a/b 为 `(B,Nx)`/`(B,Nx,Ny)`/`(B,Nx,Ny,Nz)` | ❌ `KeyError: 1`，三个 backend 全失败 | legacy grid 路径未接好 |
| `geomloss.ImagesBarycenter(measures, weights, blur=0, p=2, scaling_N=10, backward_iterations=5)` | legacy 图像 barycenter | ❌ `TypeError: list indices must be integers or slices, not tuple` | legacy 图像路径未接好 |

- `[源码]` `grid.py` 里有完整的工具函数实现（`C_transform`/`softmin_grid`/`pyramid`/`upsample`/`log_dens`，
  依赖 keops 做 1D/2D/3D 可分离卷积），仅入口 `solve_grid`/`barycenter_grid` 坏桩（缺 import + 缺实现体）。
  即"底层工具有，入口没接"。

### 3.6 实测可用性汇总（0.3.1，pykeops 已装，三 backend 可用）

| 路径 | API | 可用 |
|---|---|---|
| 点云 / 标量 loss | `SamplesLoss(backend="tensorized"\|"online"\|"multiscale"\|"auto")` | ✅ |
| 点云 / 对偶势 | `SamplesLoss(potentials=True, ...)` | ✅ |
| 点云 / ot 新 API | `ot.solve_sample(max_iter=N)` | ✅（装 pykeops 后 cost 走 lazy） |
| 点云 / ot 批 | `ot.solve_sample_batch` | ❌ 未实现桩 |
| 显式矩阵 | `ot.solve(C, reg, max_iter=N)` | ✅ |
| 批矩阵 | `ot.solve_batch(C, reg, max_iter=N)` | ✅（value 返回 (B,)） |
| barycenter（矩阵/点云/grid） | `ot.barycenter*` | ❌ 全坏桩 |
| grid / legacy | `geomloss.sinkhorn_divergence` | ❌ KeyError |
| grid / ot | `ot.solve_grid` | ❌ NameError 坏桩 |
| 图像 barycenter / legacy | `geomloss.ImagesBarycenter` | ❌ TypeError |
| image-native 类 | `ImagesLoss` / `VolumesLoss` | ❌ 0.3.1 代码未实现（官网广告） |

### 3.7 官方 API 页（pytorch-api.html）错漏清单 `[实测 + 源码]`
以 0.3.1 实测 + 源码为准，官方 API 页（`[官网API]`）的错与漏：
- **漏文档**：`sinkhorn_divergence`、`ImagesBarycenter`（顶层存在，坏）、`solve_sample_batch`（未实现桩）、
  `barycenter`/`barycenter_grid`/`barycenter_sample`（坏桩）、`OTResult` 抽象基类、`LinearOperator`。
- **未记录的坑**：`solve`/`solve_sample` 的 `max_iter` 默认 None 触发 ValueError（需显式传）；
  `solve_sample_batch` 的 `reg=0` 默认且整体未实现；`solve_sample` 的 `blur`/`reach` 冗余参数；
  `OTResultSample` 只支持非 batch。
- **属性细节偏差**：API 页称 `OTResultMatrix` 无 `potential_aa/bb`——实测是 property 存在但运行时 ValueError；
  `a_to_b`/`b_to_a` API 页归 `OTResultSample`，实测三类都有该 property 但均返回 None（基类 `return None`，子类未重写）。
- **`__all__` bug 未记录**：`geomloss.__all__` 是单元素字符串列表，`import *` 失效。
- **结论**：`[官网API]` 只能作"官方已文档部分"参考，0.3.1 完整真相以本节实测 + 源码为准。

---

## 4. backend 与 keops

`[官网首页]` 三 backend：
| backend | 需 pykeops | 性质 | 文档原话 |
|---|---|---|---|
| `tensorized` | ❌ | 朴素 O(N²)，存全矩阵，≤~5000 样本 | "A simple tensorized implementation, for small problems (< 5,000 samples)" |
| `online` | ✅ | **内存**优化（线性内存，on-the-fly map-reduce），非速度优化 | "a linear (instead of quadratic) memory footprint" |
| `multiscale` | ✅ | **速度**优化（O(N log N)，octree 块稀疏，百万级，dim≤3） | "A very fast multiscale code, which uses an octree-like structure" |

- `SamplesLoss` 的 `backend` 参数 + `ot` 子模块底层共用同一套 backend 选择（`_backends` 抽象）。
- `OTResult.lazy_plan` / `lazy_density`（KeOps LazyTensor 形式）需 pykeops；稠密 `.plan`/`.value`/`.potentials` 不需。
- `[源码]` `solve_sample` 装了 pykeops 后，cost matrix 自动走 lazy（`bk.keops_available` 为真）。

**结论**：本机 pykeops 已装 + 自检通过，**三 backend 现在都可用**。唯一真做速度复杂度优化的是
**`multiscale`**（O(N log N) 块稀疏）。`losscal` 的 `PointCloudOTLoss` 只需把 `backend="tensorized"`
改成 `"multiscale"`（或 `"auto"`）即吃上优化，无需改其它代码。

---

## 5. grid 路径结论（决定性）

- **0.3.1（最新版）的 grid 路径全坏**：`sinkhorn_divergence`（KeyError 1，所有 backend）+ `ot.solve_grid`
  （NameError: OTResult 未定义，坏桩）。**这不是 keops 问题**——即便装上 pykeops，0.3.1 的 grid
  求解器本身没接好（入口坏桩，缺 import + 缺实现体）。
- `ot.solve_grid` / `ImagesBarycenter` / `sinkhorn_divergence` 在 0.3.1 代码里存在但官方 API 页未文档，
  且 grid 实测不可用；`ImagesLoss`/`VolumesLoss` 官网首页广告但代码未实现。即 0.3.1 的图像原生 OT
  整体处于"未完成"状态。
- **0.3.1 即最新版，无升级目标**：要原生图像 grid OT，只能等 geomloss 后续发布修复版（目前没有），
  当前只能依赖点云路径（图像点云化 + SamplesLoss）。

---

## 6. 对 losscal 重构的决定性影响

1. **删 `ImageGridOTLoss`**：它调 `geomloss.sinkhorn_divergence`——API 在 0.3.1 真实存在（非"假 API"，
   修正早前判断），但 grid 路径在 0.3.1（最新版）实测坏（KeyError），且无升级目标可修复。删除决定不变，
   理由为"0.3.1 grid 坏桩，不可用"。
2. **OT 走 `SamplesLoss` 点云**：0.3.1 实测可跑，三 backend 全可用（pykeops 已装）。`PointCloudOTLoss`
   已用此 API，保留、对齐契约即可。
3. **`backend` 留可配**：默认 `"tensorized"`；将来改 `"multiscale"` 或 `"auto"` 即吃 O(N log N) 优化，
   无需新代码。
4. **grid / image-native 暂不实现**：0.3.1（最新版）grid 路径坏、`ImagesLoss`/`VolumesLoss` 未实现，
   且无升级目标版本。当前不写 grid 实现（写了也跑不了）；若未来 geomloss 发布修复版再接，统一在
   `(scalar, info)` 语义下。
5. **viz 对偶势可选**：`SamplesLoss(potentials=True)` 实测可拿 f,g，将来 OT viz 想加"transport 压力热图"
   可用；当前 viz 只给 a-b 残差即可。`[源码]` `LinearOperator`/`plan_operator`/`density_operator`
   实现完整，将来 viz 若需 transport 算子/边际也能支撑。
6. **点云新 API 备选**：`ot.solve_sample(max_iter=N)` 能顺带拿 `.plan`/`.potentials`，比 `SamplesLoss`
   信息更丰富；当前 `SamplesLoss` 标量 loss 已够，`solve_sample` 留作将来 viz 升级选项。
7. 🆕 **暴露 `blur` 而非 `reg`**：`[源码]` `SamplesLoss` 与 `solve_sample` 都支持 `blur`（几何化的"模糊半径"
   σ，`reg = p*blur**p`），比 `reg`（裸 ε）更直观。`PointCloudOTLoss` 暴露 `blur` 参数更合用户直觉。

---

## 7. 参考链接
- geomloss 首页：https://www.kernel-operations.io/geomloss/
- geomloss 安装：https://www.kernel-operations.io/geomloss/api/install.html
- geomloss PyTorch API（官方页，不完整、与代码不一致）：https://www.kernel-operations.io/geomloss/api/pytorch-api.html
- pykeops 安装：https://www.kernel-operations.io/keops/python/installation.html
- NVIDIA CUDA on WSL User Guide：https://docs.nvidia.com/cuda/wsl-user-guide/
- CUDA Toolkit 下载页（选 WSL-Ubuntu）：https://developer.nvidia.com/cuda-downloads
- Jean Feydy et al., *Interpolating between Optimal Transport and MMD using Sinkhorn Divergences*, AISTATS 2019. [Paper PDF](https://arxiv.org/abs/1810.08278) / [arXiv:1810.08278](https://arxiv.org/abs/1810.08278)
- 源码：https://github.com/jeanfeydy/geomloss （本地源码见 `.venv/lib/python3.14/site-packages/geomloss/`）
