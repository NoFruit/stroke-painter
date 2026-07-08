"""image_input.py - 目标图片导入器。

**维度范式**：图片天然三维 (C,H,W)，本项目计算空间四维 (B,C,H,W)。
导入边界（本文件）负责三维->四维：rgb 与 alpha **分离**为两个独立 tensor 输出，
不合成 4 通道。``target`` / ``canvas`` 各提供 (rgb, alpha) 两个 (1,C,H,W) tensor。

**rgb 与 alpha 分离**：rgb 延续自身语义（(1,3,H,W) RGB），alpha 单独 (1,1,H,W)。
导入图片**没有 alpha 通道**时（如 RGB），铺一个全 1 的 alpha（完全不透明）。
--PIL ``convert("RGBA")`` 原生即此语义（RGB 补不透明 alpha，已有 alpha 原样保留），
再拆 [:3]/[3:4] 分离即得。

**不做任何图像大小处理**：输入图片多大，patch 就多大。无 resize、无 target_size。

**画布为空白全 0**：canvas_rgb 全 0（黑），canvas_alpha 全 0（透明）。断点重续
canvas.png 时用源 (rgb, alpha)。

**保存默认三通道 RGB**：save_output 收 rgb，存 RGB PNG（alpha 概念全 1 不透明，不存）。

范式（本文件遵循）：**不做防御性编程**。
- target（拟合对象）：缺失 -> ``self.target_rgb/alpha = None``，不抛、不阻断。
- canvas（画布 / 断点重续载体）：缺失 -> 全 0 空白（尺寸取自 target）。
  target 也缺失且无 canvas.png -> None（下游崩溃 = 设置不对）。
- device 由 Painter 构造时传入（torch.device），用于从文件建张量（PIL->tensor、zeros canvas）。
  ImageInput 是 painter 包内唯一"凭空建张量"的组件，device 不自找。

config 全部死代码硬编码（含**绝对路径**）写在 self 上。
"""

from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image


class ImageInput:
    """目标图片导入器。所有 config 在 ``__init__`` 里死代码写死。"""

    def __init__(self, device: torch.device):
        self.device = device
        # ---- 死代码 config（绝对路径，直接可见）----
        self.target_path = "input/target.png"
        self.canvas_path = "input/canvas.png"
        # ---- 运行时状态（rgb / alpha 分离，四维）----
        self.target_rgb: Optional[torch.Tensor] = None      # (1,3,H,W) float [0,1] 或 None
        self.target_alpha: Optional[torch.Tensor] = None    # (1,1,H,W) float [0,1]（无 alpha 源补全 1）
        self.canvas_rgb: Optional[torch.Tensor] = None      # (1,3,H,W) float [0,1]，空白全 0
        self.canvas_alpha: Optional[torch.Tensor] = None    # (1,1,H,W) float [0,1]，空白全 0

    def _to_rgb_alpha(self, path: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """读图 -> (rgb(1,3,H,W), alpha(1,1,H,W)) float [0,1] on device；文件不存在返回 None。

        导入边界：PIL (H,W,C) -> torch (C,H,W) -> 计算空间四维 (1,C,H,W)，rgb/alpha 分离。
        **不做 resize**：原生尺寸即为 patch 尺寸。
        **源无 alpha 补全 1**：convert("RGBA") 后拆 [:3] / [3:4]。
        """
        import os
        if not os.path.isfile(path):
            return None
        img = Image.open(path).convert("RGBA")                   # 无 alpha -> 补不透明
        arr = np.asarray(img, dtype=np.float32) / 255.0          # [H,W,4]
        rgb = torch.from_numpy(arr[:, :, :3]).permute(2, 0, 1).contiguous().unsqueeze(0)    # [1,3,H,W]
        alpha = torch.from_numpy(arr[:, :, 3:4]).permute(2, 0, 1).contiguous().unsqueeze(0)  # [1,1,H,W]
        return rgb.to(self.device), alpha.to(self.device)

    def _blank(self, h: int, w: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """纯空画布：(rgb(1,3,H,W) 全 0, alpha(1,1,H,W) 全 0)，on device。"""
        rgb = torch.zeros(1, 3, h, w, dtype=torch.float32, device=self.device)
        alpha = torch.zeros(1, 1, h, w, dtype=torch.float32, device=self.device)
        return rgb, alpha

    @staticmethod
    def save_output(rgb: torch.Tensor) -> str:
        """把最终画布 rgb 落盘为 RGB PNG（input 旁的 output 目录，路径硬编码）。

        rgb : (1,3,H,W) RGB（累积画布）。存三通道 RGB PNG，alpha 概念全 1（不透明，不存）。
        画布尺寸即输出尺寸（target 1024×1024 -> output 1024×1024）。
        返回落盘路径。
        """
        import os
        out_path = "output/output.png"
        r = rgb.detach().cpu().float()[0].clamp(0, 1)               # (3,H,W)
        arr = (r.permute(1, 2, 0).numpy() * 255.0).astype("uint8")  # (H,W,3)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Image.fromarray(arr, mode="RGB").save(out_path)
        return out_path

    def load(self) -> "ImageInput":
        """加载 target 与 canvas 到 self（各 rgb/alpha 分离）。

        target：原生尺寸 (rgb, alpha)（无 alpha 源补全 1），缺失即 None。
        canvas：若 canvas.png 存在则 (rgb, alpha)（断点重续）；否则全 0 空白，
                尺寸取自 target。target 缺失且无 canvas.png -> None。
        """
        tgt = self._to_rgb_alpha(self.target_path)
        if tgt is not None:
            self.target_rgb, self.target_alpha = tgt
        else:
            self.target_rgb, self.target_alpha = None, None
        cv = self._to_rgb_alpha(self.canvas_path)
        if cv is not None:
            self.canvas_rgb, self.canvas_alpha = cv
        elif self.target_rgb is not None:
            H, W = self.target_rgb.shape[-2], self.target_rgb.shape[-1]
            self.canvas_rgb, self.canvas_alpha = self._blank(H, W)
        else:
            self.canvas_rgb, self.canvas_alpha = None, None       # 无 target 无 canvas -> None
        return self


__all__ = ["ImageInput"]
