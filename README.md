# stroke-painter

可微笔刷绘画：用 coarse-to-fine 优化把目标图分解为笔触序列，逐笔重建。

## 架构

```
main.py          入口：Painter(device).run()
losscal/         损失原语库（无状态）：L1 / Gradient / Area / PointCloudOT
diffbrush/       可微笔刷库：LineSquare / BezierSquare / BezierUniform
painter/         项目侧编排：CoarseToFine / ErrorMap / LossSpace / ImageInput / Painter
docs/            参考文档 + 论文文本（论文 PDF 见 REFERENCES.md 中的链接）
```

## 数据流

```
Painter.run()
  └─ coarse-to-fine 5 级金字塔
       └─ 每级 ErrorMap 矩特征初始化 -> 笔刷参数 raw
            └─ reparam(raw) -> params（合法域）
                 └─ brush.forward(params) -> (color, alpha)
                      └─ canvas = color*alpha + canvas*(1-alpha)  非预乘 blend
                           └─ LossSpace.forward(canvas, target)
                                └─ L1 + OT + Gradient + Area 加权
```

## 安装（WSL2 + RTX 3060）

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install geomloss pykeops numpy pillow matplotlib
```

> WSL2 环境下不要装含驱动的 cuda 包，只装 cuda-toolkit。

## 用法

```bash
python main.py
```

- 输入：`input/target.png`
- 输出：`output/output.png`

## 关键设计

- **笔刷输出契约**：`forward(params) -> (color(B,3,H,W), alpha(B,1,H,W))`，mainloop 负责非预乘 over 合成
- **device 语义**：Painter 是唯一 device 入口（构造必传），ImageInput 是唯一"凭空建张量"的组件，其余输入跟随
- **param_layout 驱动**：`ParamLayout` NamedTuple 声明通道布局，Painter 通过 `param_layout()` 查询通道位置，零笔刷类型硬编码
- **不做防御性编程**：缺失/错误自然崩，不 guard

## 参考论文

见 [docs/REFERENCES.md](docs/REFERENCES.md)
