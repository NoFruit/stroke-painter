"""入口：建 Painter，跑一轮 coarse-to-fine。"""

import torch

from painter.painter import Painter

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    painter = Painter(device)
    painter.run()
