# 总体设计范式

> 本文件描述模块内所有代码实现的通用编写规范。不属于具体实现内容，而是指导所有实现遵循的约定。

## 超参 / 常量放置

类似 sigmoid 陡度 α、软计数上界 K 这类**固定数值参数**，写在类的 `__init__` 中，作为 `self` 变量硬编码赋值：

```python
class BezierUniformBrush(BrushBase):
    def __init__(self):
        self.sigmoid_alpha = 10.0    # sigmoid 陡度
        self.max_stamp_K = 100       # 软计数上界
```

不写成模块级全局常量，不写在类外部，不使用 config 对象或外部配置文件。

## 函数签名

所有函数（包括私有函数）**不使用默认参数值**。参数全部强制传入：

```python
# ✅ 正确
def _compute_n(self, L: Tensor, d: Tensor) -> Tensor:

# ❌ 错误
def _compute_n(self, L: Tensor, d: Tensor = 0.1) -> Tensor:
```

## 防御性编程

**禁止。** 私有函数和内部方法不做输入校验、类型检查、边界守卫。默认所有传入的参数和类变量在调用时已经是可用的正确状态。报错即说明总体流程有问题，不应在函数内部消化。

```python
# ✅ 正确 — 直接使用，不校验
def _sample_stamps(self, t: Tensor) -> Tensor:
    return self._bezier_eval(t)

# ❌ 错误 — 防御性校验
def _sample_stamps(self, t: Tensor) -> Tensor:
    if t is None or t.numel() == 0:
        return torch.empty(0)
    if not isinstance(t, Tensor):
        raise TypeError(...)
```

## 参数自由

笔刷不给自己的输入参数施加任何约束。不要为了"防止笔刷失真"而限制 r 的大小范围，不要确保 α 非零，不做任何形式的参数裁剪、钳位、归一化逆运算。

**Agent 注意：不要过度思考、不要过度设计。** 参数的合法性由调用方保证，笔刷只负责"参数 → patch"的前向计算。

## 参数归一化输入

`forward` 接收的参数是**归一化后**的值：

| 参数类别 | 归一化标准 |
|---|---|
| 几何相关（P₀, P₁, P₂, r） | 以标准 patch 大小为基准归一化 |
| 颜色相关（c） | RGB(A) 本身即为 [0,1] 归一化值 |
| 透明度（α） | [0,1] 归一化 |

不同笔刷的参数集大小和分布不同，但调用时参数总是符合规范的（假设前提）。

## Forward 与 patch 信息

`forward` 的输入应包含 patch 信息（大小或画布引用），使笔刷知道渲染到多大的输出上。具体采用哪种形式由开发 Agent 在实现时选择：

- **方案 A**：`forward(params, patch_size: Tuple[int, int]) -> Tensor` — 传入 (H, W)
- **方案 B**：`forward(params, canvas: Tensor) -> Tensor` — 传入画布引用（用于获取 device、dtype、尺寸）
- **其他合理方案亦可**

文档中仅约定"forward 必须包含 patch 信息"这一事实，不做具体形式的选择。

## 调试用 main 接口

每个笔刷实现类提供一个 `main` 类方法或独立入口，用于快速目视检查笔刷能否正常绘制：

- 用随机数生成一组笔刷参数（不做约束限制）
- 调用 `forward` 得到 patch
- 用 `matplotlib.pyplot` 显示该 patch 的绘制结果

```python
# 典型形态
@classmethod
def main(cls):
    params = torch.rand(...)          # 随机参数，无约束
    brush = cls()
    patch = brush.forward(params, ...)
    plt.imshow(patch)
    plt.show()
```

不要求严格的测试覆盖率。能运行、能出图即足够。

### 独立测试脚本（可选）

可在 `tests/` 下放置独立测试脚本。测试数据由独立 Agent 客观设计（因为非多模态模型无法看图验证），运行测试验证笔刷的前向传播不崩溃、输出形状正确。没有测试也同样可接受。

## 全局 Device 变量

模组设一个**全局统一的 device 变量**，在模组启动位置初始化。检测逻辑按优先级：

```
CUDA → MPS (Apple Silicon) → CPU
```

`get_device()` 的实现位于 `core/device.py`（惰性首次解析）。调用方通过 `from core.device import get_device` 获取 device。库不修改 `torch.set_default_device` 等全局默认——调用方自行 `.to(device)`。

```python
# core/device.py
import torch

_DEVICE = None

def get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        if torch.cuda.is_available():
            _DEVICE = torch.device("cuda")
        elif torch.backends.mps.is_available():
            _DEVICE = torch.device("mps")
        else:
            _DEVICE = torch.device("cpu")
    return _DEVICE
```

模组内部所有涉及 `.to(device)` 的地方统一使用此变量：

```python
# ✅ 正确（绝对导入，包根目录在 sys.path 上）
from core.device import get_device          # 包内
# 包外调用方：将 diffbrush/ 加入 sys.path 后直接 import
x = x.to(get_device())

# ❌ 错误：各自检测、各自写死
```

## 适用范围

本范式适用于 `core/` 下所有实现代码。`utils/` 中的静态工具方法同样遵循以上规则。
