# 参考论文

## Birth of a Painting: Differentiable Brushstroke Reconstruction
- **作者**: Ying Jiang, Jiayin Lu, Yunuo Chen, Yumeng He, Kui Wu, Yin Yang, Chenfanfu Jiang
- **arXiv**: [2511.13191](https://arxiv.org/abs/2511.13191)
- **本项目核心参考**：笔刷参数化、coarse-to-fine 优化框架、L_app 损失设计（λ_pixel / λ_OT / λ_grad / λ_area 权重）、ErrorMap 矩特征初始化均来自此论文。

## Neural Brushstroke Engine: Learning a Latent Style Space of Interactive Drawing Tools
- **作者**: Maria Shugrina, Chin-Ying Li
- **机构**: NVIDIA, Canada
- **本项目参考**：笔刷渲染模型、stroke 参数布局设计的灵感来源。

## Jean Feydy et al., Interpolating between Optimal Transport and MMD using Sinkhorn Divergences
- **会议**: AISTATS 2019
- **arXiv**: [1810.08278](https://arxiv.org/abs/1810.08278)
- **本项目参考**：Sinkhorn 散度的数学理论、debiased OT 公式。geomloss 库即由 Feydy 作者开发，与此论文配套。
