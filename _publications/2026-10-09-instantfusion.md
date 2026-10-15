---
title: "InstantFusion: A Shared Latent Interface for Heterogeneous Image Generators"
collection: publications
permalink: /publication/2026-10-09-instantfusion
excerpt: "A lightweight shared latent interface that lets denoising hand off between frozen, independently trained image generators."
date: 2026-10-09
venue: "arXiv preprint"
status: "arXiv Preprint"
authors: "Xuanlang Dai*, Jiazi Bu*, Yujie Zhou, Yuhong Liu, Beichen Zhang, <strong>Penghui Yang</strong>, Ziyu Liu, and Yuhang Zang"
paperurl: "https://arxiv.org/abs/2610.13944"
projecturl: "https://internlm.github.io/InstantFusion/"
codeurl: "https://github.com/InternLM/InstantFusion"
modelurl: "https://huggingface.co/XuanlangDai/InstantFusion"
citation: 'Xuanlang Dai*, Jiazi Bu*, Yujie Zhou, Yuhong Liu, Beichen Zhang, <strong>Penghui Yang</strong>, Ziyu Liu, and Yuhang Zang. (2026). "InstantFusion: A Shared Latent Interface for Heterogeneous Image Generators." <i>arXiv preprint arXiv:2610.13944</i>.'
---

Xuanlang Dai\*, Jiazi Bu\*, Yujie Zhou, Yuhong Liu, Beichen Zhang, **Penghui Yang**, Ziyu Liu, and Yuhang Zang

[Paper](https://arxiv.org/abs/2610.13944){:target="_blank"} |
[Project](https://internlm.github.io/InstantFusion/){:target="_blank"} |
[Code](https://github.com/InternLM/InstantFusion){:target="_blank"} |
[Model](https://huggingface.co/XuanlangDai/InstantFusion){:target="_blank"}

InstantFusion connects the denoising trajectories of heterogeneous diffusion and
flow image generators. Model-specific Latent AutoEncoders project intermediate
states into a shared latent space, trained to align states from the same image at
matched noise levels and to synchronize their local denoising velocities.

Once trained, denoising can transition across frozen generators without a
dedicated mapping for every model pair, supporting acceleration,
multi-preference composition, cross-architecture on-policy distillation, and
cross-model composition.
