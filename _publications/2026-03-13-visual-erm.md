---
title: "Visual-ERM: Reward Modeling for Visual Equivalence"
collection: publications
permalink: /publication/2026-03-13-visual-erm
excerpt: "A multimodal generative reward model that judges vision-to-code outputs directly in the rendered visual space, giving fine-grained feedback for RL and test-time refinement."
date: 2026-03-13
venue: "NeurIPS 2026"
status: "NeurIPS 2026"
authors: "Ziyu Liu*, Shengyuan Ding*, Xinyu Fang, Xuanlang Dai, <strong>Penghui Yang</strong>, Jianze Liang, Jiaqi Wang, Kai Chen, Dahua Lin, and Yuhang Zang"
paperurl: "https://arxiv.org/abs/2603.13224"
codeurl: "https://github.com/InternLM/Visual-ERM"
dataseturl: "https://huggingface.co/datasets/internlm/VC-RewardBench"
modelurl: "https://huggingface.co/internlm/Visual-ERM"
citation: 'Ziyu Liu*, Shengyuan Ding*, Xinyu Fang, Xuanlang Dai, <strong>Penghui Yang</strong>, Jianze Liang, Jiaqi Wang, Kai Chen, Dahua Lin, and Yuhang Zang. (2026). "Visual-ERM: Reward Modeling for Visual Equivalence." <i>NeurIPS 2026</i>. arXiv:2603.13224.'
---

Ziyu Liu\*, Shengyuan Ding\*, Xinyu Fang, Xuanlang Dai, **Penghui Yang**, Jianze Liang, Jiaqi Wang, Kai Chen, Dahua Lin, and Yuhang Zang

[Paper](https://arxiv.org/abs/2603.13224){:target="_blank"} |
[Code](https://github.com/InternLM/Visual-ERM){:target="_blank"} |
[Dataset](https://huggingface.co/datasets/internlm/VC-RewardBench){:target="_blank"} |
[Model](https://huggingface.co/internlm/Visual-ERM){:target="_blank"}

Visual-ERM is a multimodal generative reward model for vision-to-code tasks such
as chart-to-code, table-to-markdown, and SVG-to-code. Instead of textual rules or
coarse embedding similarity, it compares the ground-truth image with the rendered
prediction and reports fine-grained, interpretable discrepancies.

Used as a reward in RL, it improves Qwen3-VL-8B-Instruct by **+8.4** on
chart-to-code, with gains of **+2.7** and **+4.1** on table and SVG parsing. The
paper also introduces VC-RewardBench for judging fine-grained image-to-image
discrepancies on structured visual data.
