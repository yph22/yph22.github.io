---
title: "CapRL++: Unified Reinforcement Learning with Verifiable Rewards for Dense Image and Video Captioning"
collection: publications
permalink: /publication/2026-06-08-caprl-plus-plus
excerpt: "A unified RLVR framework for dense image and video captioning, where caption quality is optimized through verifiable downstream question-answering rewards. Submitted to TPAMI."
date: 2026-06-08
venue: "arXiv preprint"
status: "Submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)"
paperurl: "https://arxiv.org/abs/2606.09393"
citation: '<strong>Penghui Yang</strong>*, Long Xing*, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Yibin Wang, Yujie Zhou, Jiazi Bu, Jianze Liang, Qidong Huang, Jiaqi Wang, Feng Wu, and Dahua Lin. (2026). "CapRL++: Unified Reinforcement Learning with Verifiable Rewards for Dense Image and Video Captioning." <i>arXiv preprint arXiv:2606.09393</i>. Submitted to TPAMI.'
---

CapRL++ extends CapRL from dense image captioning to a unified image and video
caption reinforcement learning framework. Instead of relying on fixed reference
captions, CapRL++ uses verifiable downstream question-answering rewards: a
caption is considered high quality when a text-only model can answer visual
questions from that caption alone.

The framework introduces reward components for visual utility, temporal
timestamp formatting, and length-aware regularization, enabling dense,
chronological, and information-rich captions for videos.

[Paper](https://arxiv.org/abs/2606.09393){:target="_blank"} |
[Project](https://github.com/InternLM/CapRL){:target="_blank"} |
[Model](https://huggingface.co/internlm/CapRL-Video-4B){:target="_blank"}
