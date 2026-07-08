# differential_brush_core — Harness 文档

## 模块定位

可微笔刷渲染的核心模块。实现一条"笔刷参数 → 可微渲染 → 像素梯度反传"的通路，使笔刷几何/纹理参数可通过梯度优化直接拟合目标图像。

`differential_brush_core` 是一个可微笔刷渲染模块。`core/` 下平铺所有笔刷实现与基础类，`utils/` 下放置共享数学工具。

## 阅读指引

### 给 LLM Agent

本目录下各 `.md` 文件构成一套结构化 prompt，描述模块的架构约定、数学形式、接口契约和实现规范。Agent 在参与本模块的阅读、修改、扩展前，应顺序阅读以下文档以获得完整上下文。每篇文档头部标注了它的前置依赖。

### 给人类开发者

措辞和格式同时面向人和机器。如有歧义以 LLM Agent 阅读优先级为准——文档是为 Agent 能精确理解而写的，人类读起来也应清晰。

## 文档路径即代码蓝图

`doc/` 下的目录结构镜像代码目录结构。每一篇 `.md` 文件标注了建议的代码路径（`**代码路径建议：**`）。实作时应尽量对齐此结构。

```
doc/                          ← 入口
├── README.md
├── design_paradigm.md        ← 总体设计范式
├── core/
│   ├── device.md             ← 全局统一 device 管理
│   └── base/
│       └── brush_base.md     ← BrushBase 抽象基类契约
├── implementations/
│   ├── bezier_uniform_brush.md  ← BezierUniformBrush 详情
│   ├── bezier_square_brush.md   ← BezierSquareBrush 详情
│   └── line_square_brush.md     ← LineSquareBrush 详情
└── utils/
    └── README.md             ← 通用数学工具
```

## 文档清单

- [总体设计范式](design_paradigm.md) — 超参放置、函数签名、防御性编程等通用编码规范
- [全局 Device 管理](core/device.md) — 统一 device 检测与使用
- [BrushBase 契约](core/brush_base.md) — 笔刷抽象基类接口定义
- [BezierUniformBrush 详情](core/bezier_uniform_brush.md) — 单色均匀粗细贝塞尔笔刷
- [BezierSquareBrush 详情](core/bezier_square_brush.md) — 方头贝塞尔笔刷
- [LineSquareBrush 详情](core/line_square_brush.md) — 方头直线笔刷
- [utils 工具](utils/README.md) — 通用数学函数
