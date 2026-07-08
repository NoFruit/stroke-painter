# BrushBase — 笔刷抽象基类

**代码路径建议：** `core/base/brush_base.py`

## 定位

所有笔刷实现必须遵循的抽象契约。

## 接口

```python
class BrushBase(ABC):
    @abstractmethod
    def forward(self, params: Tensor, *args, **kwargs) -> Tensor:
        """归一化笔刷参数 → 可微 patch

        params 为归一化后的参数张量（归一化标准见 design_paradigm.md）。
        实现时必须包含 patch 信息（大小或画布引用），
        具体形式由实现者选择。

        Args:
            params: 归一化后的笔刷参数张量，形状及含义由子类定义。
            *args, **kwargs: 实现类自定义的额外输入（如 patch 信息）。

        Returns:
            渲染 patch，形状 (B, C, H, W)，梯度可通到 params。
        """
        ...
```

## 继承约定

- 子类必须实现 `forward`。
- 子类可扩展额外方法，但不得改变 `forward` 的"参数 → patch"语义。
- 子类不得对输入参数施加任何约束（钳位、裁剪、范围校验等）。参数自由，合法性由调用方保证。
