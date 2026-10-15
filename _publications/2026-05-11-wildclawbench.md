---
title: "WildClawBench: A Benchmark for Real-World, Long-Horizon Agent Evaluation"
collection: publications
permalink: /publication/2026-05-11-wildclawbench
excerpt: "A native-runtime benchmark of 60 human-authored, bilingual, multimodal long-horizon tasks that evaluates AI agents end-to-end inside real CLI agent harnesses."
date: 2026-05-11
venue: "arXiv preprint"
status: "arXiv Preprint"
authors: "Shuangrui Ding*, Xuanlang Dai*, Long Xing*, Shengyuan Ding, Ziyu Liu, Jingyi Yang, <strong>Penghui Yang</strong>, Zhixiong Zhang, Xilin Wei, Xinyu Fang, Yubo Ma, Haodong Duan, Jing Shao, Jiaqi Wang, Dahua Lin, Kai Chen, and Yuhang Zang"
paperurl: "https://arxiv.org/abs/2605.10912"
projecturl: "https://internlm.github.io/WildClawBench/"
codeurl: "https://github.com/InternLM/WildClawBench"
dataseturl: "https://huggingface.co/datasets/internlm/WildClawBench"
citation: 'Shuangrui Ding*, Xuanlang Dai*, Long Xing*, Shengyuan Ding, Ziyu Liu, Jingyi Yang, <strong>Penghui Yang</strong>, Zhixiong Zhang, Xilin Wei, Xinyu Fang, Yubo Ma, Haodong Duan, Jing Shao, Jiaqi Wang, Dahua Lin, Kai Chen, and Yuhang Zang. (2026). "WildClawBench: A Benchmark for Real-World, Long-Horizon Agent Evaluation." <i>arXiv preprint arXiv:2605.10912</i>.'
---

Shuangrui Ding\*, Xuanlang Dai\*, Long Xing\*, Shengyuan Ding, Ziyu Liu, Jingyi Yang, **Penghui Yang**, Zhixiong Zhang, Xilin Wei, Xinyu Fang, Yubo Ma, Haodong Duan, Jing Shao, Jiaqi Wang, Dahua Lin, Kai Chen, and Yuhang Zang

[Paper](https://arxiv.org/abs/2605.10912){:target="_blank"} |
[Leaderboard](https://internlm.github.io/WildClawBench/){:target="_blank"} |
[Code](https://github.com/InternLM/WildClawBench){:target="_blank"} |
[Dataset](https://huggingface.co/datasets/internlm/WildClawBench){:target="_blank"}

WildClawBench evaluates whether AI agents can complete realistic long-horizon work
in the runtimes where they are deployed. Each of its 60 tasks runs in a
reproducible Docker container hosting an actual CLI agent harness (OpenClaw,
Claude Code, Codex, or Hermes Agent) with access to real tools, averaging roughly
8 minutes and over 20 tool calls.

Across 19 frontier models in the paper, the best reaches only **62.2%** overall
under OpenClaw, and switching harness alone shifts a single model by up to
**18 points**.
