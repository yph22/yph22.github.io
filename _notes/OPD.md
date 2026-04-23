---
published: true
title: "重温OPD（On-Policy Distillation）"
collection: notes
date: 2026-04-23
excerpt: "markdown格式预览"
---

# 重温 On-Policy Distillation

## （1）从 SFT / SeqKD 到 OPD 的第一性梳理

> 这篇是 OPD 学习笔记第一篇：先从 LLM Distillation 讲起，作为 OPD 两条路线（GKD-style / PG-style）的前置铺垫。

### 为什么想重学 OPD？

最近在知乎看到一篇讨论 OPD 两种技术路线（PG-style 和 GKD-style）的文章（文末有链接），才意识到自己之前对 OPD 的理解还不够系统和深入，很多地方只是停留在概念层面。  
所以打算重新梳理 OPD 的发展脉络，也顺带复习 RL，并把学习过程整理成系列笔记。  
本文参考了若干公开资料，也加入了个人理解；若有不准确之处，欢迎交流指正。

### Part 1：先从 LLM Distillation 说起

在 LLM 训练中，SFT 是最常见的方法之一。严格来说，SFT 不一定都属于 Distillation：  
若训练序列来自人工标注，本质为人类监督学习；若序列来自更强的 teacher model 采样，才属于典型蒸馏（student-teacher）语境。

### 1）SFT：hard label 的 token 学习

给定前缀序列

$$
s = (t_1, t_2, \dots, t_{n-1}),
$$

SFT 的目标是在下一位置生成数据中的真实 token $t_n$。  
这等价于把目标分布视为 one-hot：$t_n$ 概率为 1，其它 token 概率为 0。对应损失为

$$
L_t = -\log \pi_{\theta}(t_n \mid s).
$$

其中 $\pi_{\theta}$ 是 student 的参数化策略（条件概率分布）。

### 2）SeqKD：soft label 的分布学习

SeqKD 同样由 teacher 采样完整序列，但 teacher 不只提供“最终采样 token”，还提供该位置完整词表分布。  
因此 student 的学习目标不再是 one-hot，而是贴近 teacher 的全分布：

$$
L_t = -\sum_{i=1}^{V} \pi_{\text{teacher}}(t_i \mid s)\,
\log \pi_{\theta}(t_i \mid s).
$$

该目标本质上与 teacher/student 在该位置分布之间的 KL 相关（称为 Forward KL）。

### 3）SFT 与 SeqKD 的关系

SFT 可以看作 SeqKD 的退化情况：

- 当 $t_i = t_n$ 时，$\pi_{\text{teacher}}(t_i \mid s) = 1$；
- 否则，$\pi_{\text{teacher}}(t_i \mid s) = 0$。

但真实 teacher 分布通常并非严格 one-hot。  
因此 SeqKD 相比 SFT 提供更丰富的 soft label 信息：不仅告诉 student 哪个 token 对，也告诉它其它候选 token 的相对概率。

### 4）从 SeqKD 到 OPD 的直觉

在 RL 大规模用于 LLM 之前，SFT 和 SeqKD 可能都是核心范式（今天 SFT 依然很重要）。  
它们共同点是：都在 token-level 提供监督，并可通过梯度下降直接优化参数。

顺着 SeqKD 的形式看，OPD 与其高度对称：

- 都需要比较 teacher/student 在序列各位置的词表分布；
- 都在 token-level 计算损失；
- 关键差异在于该序列由谁采样（teacher 或 student）。

由此可自然得到 OPD 的一条实现路径：**GKD-style**（分布匹配、可导优化）；  
另一条路径是 **PG-style**（通过 Policy Gradient 更新 student），这条路径从 RL 的视角引入或许更加自然，留作下节细讲。

### 5）补充：SFT 和 SeqKD 的特点和局限性

关于 SFT 和 SeqKD，有不少 paper 讨论它们在算法层面的效果和局限性（如泛化性、灾难性遗忘等等）。  
二者都属于 off-policy，即采样序列来自 teacher，当然更直观的解释（为什么不如 on-policy）可以理解为 student 在学习的过程中，我们不知道它学的怎么样；而 on-policy 由于每次更新时用到的采样序列都是来自最新（或最近）的 student，因此实际上提供了 student 在学习过程中的逐步变化。

另外，这里再提醒一下，SFT 之所以应用更广泛，除了实现上比较简单，另一个特点是 SFT 可以用在 black-box model 的蒸馏上。  
例如一些最先进的闭源模型，我们没法获取 token 的 logit，只能把完整的序列视作 one-hot 分布让 student 拟合。  
此外，也有工作把 on-policy 用在了 black-box 的蒸馏上（如 general adversarial distillation），感兴趣的朋友可以自行了解。

### 下一篇预告

下一篇将重点对比 GKD-style 与 PG-style 的差异和联系。

**参考链接（知乎原文）：** <https://zhuanlan.zhihu.com/p/2027548813129267030>

---

## （2）从 RL 到 OPD 的直觉联系

### 前言

上一节我们从 SFT、SeqKD 出发，探讨了几种 LLM Distillation 之间的联系，并自然引出 OPD 可以视为 SeqKD 的镜像版本，也即得到 GKD-style 的 OPD。关于它的具体定义和损失函数，将留到后文说明。

本节主要探讨一下，同样作为 on-policy 的学习手段，OPD 和 RL 之间存在什么关联？毕竟 RL 相对热门的时间更早一些，大家可能对它更熟悉。  
从 RL 的角度审视 OPD，可以得到 OPD 的另一种技术路径：PG（Policy Gradient）-style OPD。

### Part 1：先从 RL 说起

从去年开始，RLVR 在社区大规模兴起，时至今日仍然是 LLM post-training 中十分重要的一环。  
它的一个重要特点是鼓励 model 自己的探索，而不像 SFT 那样对 model 每一次生成 next token 都加以强监督；  
换句话说，RLVR 只关心 model 生成的完整序列好不好，不关心这个序列的每一处 token 是否生成的合理。  
这可以极大地缓解由 SFT 可能带来的“死记硬背”的泛化性差的问题。

RL 作为一种 on-policy 的手段，每次更新时所用到的序列都是由（最新的，或者最近的）model 自己采样得到的。  
在上一节我们提到，这种 on-policy 的手段将有助于在 LLM 的学习过程中提供【实时】的检测效果，即通过每一次实时的采样，我们都能获取到 model 最新的学习情况，进而提供更精准的指导信号。  
作为代价，RL 在训练中需要增加 model 实时的采样（而 SFT/SeqKD 等都是提前把序列准备好了），这显著拖慢了训练速度。

现在让我们再次回顾一下常见 RLVR 的一个核心特点：只关心【完整序列】的好坏，不关心【每个 token】的好坏。  
上面说到这样做的好处是防止 model 死记硬背，同时可以激励 model 自由探索；  
但是也可以说它所提供的指导信号太稀疏，这样的指导到底比提供更细粒度的指导是好是坏，我们不去探讨。  
此外，实际上除了主流的 RLVR，在早期传统 PPO 算法中也可以设计一个专门用来提供逐 token 监督信号的模型，但是由于训练时 model 太多、成本太高等因素，现在已经很少见了。

实际上除了训练效果上的讨论，RL 与 SFT 等还有一个重要区别是，如何根据损失函数去优化模型参数？  
这涉及到 loss 是否直接针对参数可导的问题，需要仔细说说，方便后面引出 PG-style 的 OPD。

### 1）RL：通过 Policy Gradient 更新 model 参数

在上一节中，我们提到了 SFT 和 SeqKD 的损失函数（前者可以视为后者的一个退化），例如对于 SeqKD（这里给出完整的 KL 散度版本，和上一节求梯度是等价的，只是差了个常数项）：

$$
L_t=D_{\mathrm{KL}}\!\left(\pi_{\text{teacher}}(\cdot\mid s)\,\|\,\pi_\theta(\cdot\mid s)\right)
=
\sum_{i=1}^{V}\pi_{\text{teacher}}(t_i\mid s)
\log \frac{\pi_{\text{teacher}}(t_i\mid s)}{\pi_\theta(t_i\mid s)}.
$$

通过这个 Loss，我们要去优化的是 student model 的权重参数 $\theta$，  
最终达到的目标是在这个 token 位置处，student 生成词表中每一个 token 的概率 $\pi_{\theta}(t_i \mid s)$，都尽可能与 teacher 保持一致。

注意，不要忘记我们的优化对象——model 的权重参数 $\theta$，很明显，上面这个定义在 token 上的 $L_t$，很明显是直接针对 $\theta$ 可导的，根本原因是这个 token 处的监督信号是针对 **model 概率分布 $\pi_\theta$** 的函数，所以可以由链式法则将梯度一路传回去。

那么切换到 RL 场景，我们前面提到，无论是序列级别的监督信号（以 RLVR 为典型），还是 token 级别的监督信号（以 PPO 为代表），  
它们的损失（或者称为奖励），都是定义在 **【采样出来的 token】** 上的（前者通过广播或平均机制，从序列级信号得到 token 级的信号），而不是采样时的概率分布。

需要特别注意的点在于，model 通过权重参数从输入一路前向传播到输出最终的词表概率分布，都是可导的（这里只讨论 Dense model）。  
而采样操作：从最终的词表概率分布选出某一个 token，这个过程是不可导的（例如 argmax 的不可导性）。  
但由于 RL 中的监督信号都是定义在【采样出来的 token（不妨称为 $y$）】上的，因此梯度从监督信号往回传到 $y$ 之后就被截断掉了，因为无法定义一个采样操作与其概率分布的导数（非光滑映射）。

用符号来表示的话，RL 的计算图

> $\theta$ $\leftrightarrow$ $\pi_\theta$ $\to$ sample $y$ $\leftrightarrow$ $R(y)$

在 $\pi_\theta$ $\to$ sample $y$ 反向传播时梯度会断掉（即 $\frac{\partial y}{\partial \pi_\theta}$ 不存在）。这里 $R(y)$ 即表示我们的监督信号（或称为奖励）是定义在被采样出来的 token 上的，而前面的 SeqKD 的监督信号是直接定义在采样时的概率分布上的。

既然 RL 的信号梯度无法直接回传到参数 $\theta$ 上，那么我们又该如何根据这个信号去更新 model 参数呢？核心思想是 **虽然单次 sample 不可导，但期望（多次采样）是可导的**。  
具体来说，我们不直接去优化 $R(y)$，而是去优化 $R(y)$ 的期望：

$$
  J(\theta) = E_{y \sim \pi_\theta}[R(y)]=\sum_y \pi_\theta(y)R(y)
$$

这里 $\pi_\theta(y)$ 就表示在当前输入下，model 采样生成 $y$ 的概率。这样转为求期望，我们就从 **【采样出来的 token】** 引入了 **【采样时的概率分布】**，  
进而实现梯度回传（此时可导）：

$$
  \nabla_\theta J(\theta) = \sum_{y} \nabla_\theta \, \pi_\theta(y) \, R(y)
$$

为了更清晰的展开梯度表达式，这里需要用到对数导数技巧：

$$
  \nabla_\theta \, \pi_\theta(y) = \pi_\theta(y) \, \nabla_\theta \log \pi_\theta(y)
$$

代入有

$$
  \nabla_\theta J(\theta) 
  = \sum_{y} \pi_\theta(y) \, R(y) \, \nabla_\theta \log \pi_\theta(y) 
  = \mathbb{E}_{y \sim \pi_\theta} \left[ R(y) \, \nabla_\theta \log \pi_\theta(y) \right]
$$

这就是 **Policy Gradient**，它避开了不可导的 $\frac{\partial y}{\partial \theta}$，转化为 $\nabla_\theta \log \pi_\theta(y)$，就能正常使用梯度下降去更新参数 $\theta$ 了。

### 2）从 RLVR 到 OPD：稀疏奖励的改进

前面的推导解决了 RL 如何优化 model 参数的问题，现在让我们回到 RL（后面均指 RLVR）的第一个问题：奖励稀疏。  
也就是前面所说，我们提供的信号是针对完整序列而言，而不关心每一个 token 的好坏。  
那我们能不能像 SeqKD/SFT 那样，也去关注每个 token 的好坏呢？  
我们暂且不讨论这样做最终导致的优劣，但从信息量来说明显这样提供的指导信号是更丰富的。这就是 OPD 的 **PG-style** 的实现。

具体来说，在上面求 RL 的梯度时，把 $R(y)$ 直接换成 student 与 teacher 生成当前 token $y$ 的概率比值，即

$$
  \nabla_\theta J(\theta) 
  = \mathbb{E}_{y \sim \pi_\theta} \left[ \log \frac{\pi_\theta(y)}{\pi_{teacher}(y)} \, \nabla_\theta \log \pi_\theta(y) \right]
$$

这里需要注意的是，$\log \frac{\pi_\theta(y)}{\pi_{teacher}(y)}$ 本身是会引入关于 $\theta$ 的梯度项的，但在这里我们相当于施加了 **stop-gradient** 操作，  
即梯度完全由最右侧 $\nabla_\theta \log \pi_\theta(y)$ 产生，而前面的 $\log \frac{\pi_\theta(y)}{\pi_{teacher}(y)}$ 只提供一个标量的更新力度。

另外，这里（包括前面 RL 的实现）的期望在工程上不可直接求解，因此使用 Monte Carlo 采样去近似。  
简单来说就是多次采样取均值，不过需要稍微提醒一下，这里说的多次采样都是针对序列级别，例如给定一个 prompt，生成多个响应序列，  
而上面为了直观对比，推导都是基于 token 级别的 loss，大家也可以换成 sequence 级别的 loss（利用概率乘积即可从 token 得到完整序列）。

#### 小结：PG-style OPD vs RLVR

最后小结一下，在 RLVR 中，$R(y)$ 本质上是基于完整序列的信号，然后通过广播折算到每个 token 上，也就是说所有 token 的贡献都是一样的。  
而 OPD 则使用 $\log \frac{\pi_\theta(y)}{\pi_{teacher}(y)}$ 作为逐 token 各自的指导更新力度，因此相比于 RLVR 可以获得更 dense 的信号。

### 下一篇预告

下一篇将说说 PG-style 和 GKD-style 二者对于 model 更新上的差别，同时也总结一些相关工作，例如关于 KL 的选择（reverse vs forward、top-k、k1/k2/k3 近似）等等。

---

## （3）GKD-style 的更新模式

### 前言

第一节我们从 SFT 和 SeqKD 出发，得到了 GKD-style 的 OPD，并指出它实际上就是 SeqKD 的镜像版本；  
第二节我们从 RL 出发，得到了 PG-style 的 OPD，并指出它可以视为一种提供密集信号的 RLVR。  
这一节我们先给出 GKD-style OPD 更正式的数学定义，并通过梯度分析它是如何促进 student 更新参数的，  
从动力学优化的角度看看它到底实现了怎样的效果，同时为后面和 PG-style 的对比做个铺垫。

### GKD-style OPD

回顾 SeqKD 的 Loss 函数，这里直接给出 Sequence 版本：

$$
  L_{SeqKD}(\theta) =\sum_t D_{KL}(\pi_{teacher}(\cdot | s_t), \pi_{\theta}(\cdot | s_t))
$$

这里 $t$ 表示的是当前序列 $\tau$ 的第 $t$ 个 token 位置，$s_t$ 表示当前位置 $t$ 时的输入前缀序列（用 $a_t$ 表示在第 $t$ 个 token 位置处实际采样的 token），即

$$
  s_t=(token_{prompt}, a_1, a_2 \dots a_{t-1})
$$

同时要注意，上面使用的 $\pi(\cdot | s_t)$（表示的是一个概率分布），而不是 $\pi(a_t | s_t)$（选择某个 token 的概率值）。  
完整的展开即有

$$
  L_{SeqKD}(\theta) =\sum_t \sum_v \pi_{teacher}(a_v | s_t) \log \frac{\pi_{teacher}(a_v | s_t)}{\pi_{\theta}(a_v | s_t)}
$$

上式的 $\sum_t$ 表示对一个 sequence 的每一个位置的 token 的 loss 求和，  
$\sum_v$ 表示在每一个 token 位置 $t$ 处，对词表中所有 token 的概率信号求和（实际上也就是完整 KL 的展开式，它是定义在每个 token 上的）。

这里还有一个需要注意的点是，SeqKD 中的序列（或者说序列中的每一个 token）是由 teacher 采样而来，  
这被称为 **forward-KL**，即有

$$
  \tau \sim \pi_{teacher}, a_t \sim \pi_{teacher}(\cdot | s_t)
$$

现在我们直接交换 KL 散度公式中 teacher 和 student（即 $\theta$）的位置，是不是就能得到 GKD-style OPD 呢？

$$
  L_{OPD}(\theta) =\sum_t D_{KL}(\pi_{\theta}(\cdot | s_t), \pi_{teacher}(\cdot | s_t)) =\sum_t \sum_v \pi_{\theta}(a_v | s_t) \log \frac{\pi_{\theta}(a_v | s_t)}{\pi_{teacher}(a_v | s_t)}
$$

这样做的话，好像只是在计算 KL 时换了个算法，但这里忽略了一点：on-policy 意味着轨迹是由 student 自己采样的，因此还有

$$
  \tau \sim \pi_{\theta}, a_t \sim \pi_{\theta}(\cdot | s_t)
$$

实际上这就是 **reverse-KL** 的实现，它不单单只是在计算 loss 时把 SeqKD 中的 student 和 teacher 换了个位置，更重要的是采样来源也变了。这样做会带来一个与 SeqKD 不同的后果：  
SeqKD 采用 forward KL 追求的是 student 的词表概率分布尽可能和 teacher 完全对齐（mode-covering），  
而 OPD 采用 reverse KL 追求的是一种 mode-seeking，具体见下文。

#### GKD-style 的梯度

我们推导一下 OPD 的梯度，将单步 reverse KL 展开：

$$
\ell_t(\theta)
=
\mathrm{KL}\!\left(
\pi_{\theta}(\cdot \mid s_t)\,\|\,\pi_{\text{teacher}}(\cdot \mid s_t)
\right)
=
\sum_{v}
\pi_{\theta}(a_v \mid s_t)
\log
\frac{\pi_{\theta}(a_v \mid s_t)}{\pi_{\text{teacher}}(a_v \mid s_t)}.
$$

对参数 $\theta$ 求导（此处省略过程，很简单），可得：

$$
\boxed{
\nabla_{\theta}\,\ell_t(\theta)
=
\sum_v
\left(
\log \frac{\pi_{\theta}(a_v \mid s_t)}{\pi_{\text{teacher}}(a_v \mid s_t)}
+ 1
\right)
\nabla_{\theta}\,\pi_{\theta}(a_v \mid s_t)
}
$$

因此最终完整的 GKD-style OPD 梯度为：

$$
\boxed{
\nabla_{\theta}\,L_{\text{OPD}}(\theta)
=
\sum_t \sum_v
\left(
\log \frac{\pi_{\theta}(a_v \mid s_t)}{\pi_{\text{teacher}}(a_v \mid s_t)}
+ 1
\right)
\nabla_{\theta}\,\pi_{\theta}(a_v \mid s_t)
}
$$

这个式子的关键含义是：**在每个访问到的前缀上，词表中的所有 token 都会同时产生梯度信号。**  
因此，GKD-style 优化的不是某个 sampled token 的概率，而是 student 在该位置上的 **整个 next-token 分布**。  
这就是所谓的“推动完整词表概率对齐”。

需要特别注意的是，这里不能简单地把

$$
\left(
\log \frac{\pi_{\theta}(a_v \mid s_t)}{\pi_{\text{teacher}}(a_v \mid s_t)}
+ 1
\right)
$$

理解为“第 $v$ 个 token 单独的更新方向”。  
因为 $\nabla_{\theta}\,\pi_{\theta}(a_v \mid s_t)$ 是参数空间中的高维向量，不同 token 的梯度彼此耦合；再加上 softmax 的归一化约束，  
**一个 token 概率的变化会同时影响其他 token 的概率**。  
因此，如果想分析“某个 token 会被推高还是压低”，更合适的对象不是参数梯度，而是 **logit 梯度**。

### 从 logit 角度理解 mode-seeking 下的概率分布对齐

下面我们先公式推导，然后再列举实际例子，看看 reverse-KL 会导致 student 具体怎么更新。

**公式分析**

设当前位置 student 的 logits 为 $z_v$（即在最终 softmax 归一化得到概率之前的值），对应概率为

$$
p_v = \pi_{\theta}(a_v \mid s_t), \qquad
q_v = \pi_{\text{teacher}}(a_v \mid s_t).
$$

则单步 reverse KL 为

$$
\ell_t = \sum_v p_v \log \frac{p_v}{q_v}.
$$

对某个 token $a_j$ 的 logit $z_j$ 求导，可以得到（这一步相对比较复杂，后面让 GPT 写了一个详细过程，此处只给最终结果）：

$$
\boxed{
\frac{\partial \ell_t}{\partial z_j}
=
p_j
\left(
\log \frac{p_j}{q_j}
-
\sum_v p_v \log \frac{p_v}{q_v}
\right)
}
$$

也就是

$$
\boxed{
\frac{\partial \ell_t}{\partial z_j}
=
p_j
\left(
\log \frac{p_j}{q_j}
-
\mathrm{KL}(p\|q)
\right)
}
$$

仔细看这个式子，左侧 $\frac{\partial \ell_t}{\partial z_j}$ 表示当前 student 针对 $z_j$ 的更新方向，即让 $z_j$ 变大或者变小（对应概率 $p_j$ 变大变小）。  
而右侧括号内，第一项 $\log \frac{p_j}{q_j}$ 可以视为针对词表中的这个 token $a_j$，student 和 teacher 的不匹配程度；第二项 $\mathrm{KL}(p\|q)$ 可以视为针对词表中的所有 token（即完整的词表概率分布），student 和 teacher 的不匹配程度。

也就是说，括号内可以视为：**当前 token 的不匹配程度与全局 token 的不匹配程度的差异。**  
我们接着看完整的梯度下降更新过程为

$$
z_j \leftarrow z_j - \eta \frac{\partial \ell_t}{\partial z_j}.
$$

因此：

- 若 $\log \frac{p_j}{q_j} < \mathrm{KL}(p\|q)$，则 $\frac{\partial \ell_t}{\partial z_j} < 0$，梯度下降会 **提高** 该 token 的 logit；
- 若 $\log \frac{p_j}{q_j} > \mathrm{KL}(p\|q)$，则 $\frac{\partial \ell_t}{\partial z_j} > 0$，梯度下降会 **降低** 该 token 的 logit。

也就是说：

- 如果某个 token 的不匹配程度高于平均水平，梯度为正，梯度下降会压低它；
- 如果某个 token 的不匹配程度低于平均水平，梯度为负，梯度下降会抬高它。

这说明 reverse KL 的行为并不是“逐 token 地把 student 的概率值硬拉到 teacher 的对应值”，  
即针对某个 token，如果 teacher 的生成概率大于 student，就提升 student 的概率；如果 teacher 的生成概率小于 student，就降低 student 的概率。  
而是更像一种 **对整个分布形状的重塑**：  
它会强烈打压 student 过度押注、而 teacher 不认可的高概率模式，同时把概率重新分配到 teacher 支持的候选上。  
这也正是 reverse KL 常被称为 *mode-seeking* 的原因。

**具体例子**

假设当前前缀下 student 和 teacher 的分布分别为：

$$
p=(0.2,\;0.3,\;0.5), \qquad q=(0.7,\;0.2,\;0.1).
$$

先计算各 token 的 log-ratio：

$$
\begin{aligned}
r_1 &= \log \frac{0.2}{0.7} \approx -1.25,\\
r_2 &= \log \frac{0.3}{0.2} \approx 0.41,\\
r_3 &= \log \frac{0.5}{0.1} \approx 1.61.
\end{aligned}
$$

再计算当前的 reverse KL：

$$
\mathrm{KL}(p\|q)
=
0.2 \log \frac{0.2}{0.7}
+
0.3 \log \frac{0.3}{0.2}
+
0.5 \log \frac{0.5}{0.1}
\approx 0.68.
$$

于是各个 token 的 logit 梯度符号为：

$$
\frac{\partial \ell_t}{\partial z_j}
=
p_j (r_j - 0.68).
$$

逐一分析：

- **$t_1$**：$r_1=-1.25 < 0.68$，因此 $\frac{\partial \ell_t}{\partial z_1}<0$，梯度下降会 **提高** $t_1$ 的 logit，从而提高其概率；
- **$t_2$**：$r_2=0.41 < 0.68$，因此 $\frac{\partial \ell_t}{\partial z_2}<0$，梯度下降也会 **提高** $t_2$ 的 logit；
- **$t_3$**：$r_3=1.61 > 0.68$，因此 $\frac{\partial \ell_t}{\partial z_3}>0$，梯度下降会 **降低** $t_3$ 的 logit，从而压低其概率。

可见，reverse KL 的行为并不是简单地“student 概率高于 teacher 就压低，低于 teacher 就抬高”。  
例如在这个例子里，$t_2$ 虽然 student 的概率 $0.3$ 略高于 teacher 的 $0.2$，但它仍然会被 **抬高**，因为当前 student 最大的问题是对 $t_3$ 赋予了过高概率；reverse KL 会优先压低这种 teacher 明显不认可的主峰，并把概率质量重新分配到 teacher 更支持的其它 token 上。

#### SeqKD 采用的 forward-KL

同样都是实现“完整词表概率对齐”，相比于 GKD-style OPD 常采用的 reverse-KL，SeqKD 中更自然的目标通常是 forward-KL：

$$
\ell_t^{\text{fKL}}
=
\mathrm{KL}\!\left(
q \,\|\, p
\right)
=
\sum_v q_v \log \frac{q_v}{p_v},
$$

其中

$$
p_v = \pi_{\theta}(a_v \mid s_t), \qquad
q_v = \pi_{\text{teacher}}(a_v \mid s_t).
$$

由于 teacher 分布 $q$ 不依赖于 student 参数，因此 forward-KL 也可以写成

$$
\ell_t^{\text{fKL}}
=
\underbrace{\sum_v q_v \log q_v}_{\text{与 student 无关的常数}}
-
\sum_v q_v \log p_v.
$$

因此，最小化 forward-KL 等价于最小化 teacher 软标签下的交叉熵：

$$
\ell_t^{\text{fKL}}
\equiv
-\sum_v q_v \log p_v.
$$

**对 logit 的梯度**

设 student 在该位置的 logits 为 $z_j$，则 softmax 概率为

$$
p_j = \frac{e^{z_j}}{\sum_k e^{z_k}}.
$$

forward-KL 对 logit 的梯度有一个非常经典且简洁的结果：

$$
\boxed{
\frac{\partial \ell_t^{\text{fKL}}}{\partial z_j}
=
p_j - q_j
}
$$

因此梯度下降更新为

$$
z_j \leftarrow z_j - \eta (p_j - q_j).
$$

于是可以直接得到：

- 若 $p_j > q_j$，则 $\frac{\partial \ell_t^{\text{fKL}}}{\partial z_j} > 0$，梯度下降会 **降低** 该 token 的 logit，从而压低其概率；
- 若 $p_j < q_j$，则 $\frac{\partial \ell_t^{\text{fKL}}}{\partial z_j} < 0$，梯度下降会 **提高** 该 token 的 logit，从而提升其概率；
- 若 $p_j = q_j$，则该 token 在当前位置上达到局部匹配，不再产生 logit 梯度。

这说明 forward-KL 的行为非常直接：  
它会 **逐 token 地** 把 student 的概率往 teacher 的对应概率上拉。也就是说，对每个 token 来说：

- teacher 给得比 student 高，就把 student 往上抬；
- teacher 给得比 student 低，就把 student 往下压。

因此，forward-KL 更像是一种 **逐坐标的分布校正**。

**具体例子**

仍然使用前面的同一个例子：

$$
p=(0.2,\;0.3,\;0.5), \qquad q=(0.7,\;0.2,\;0.1).
$$

则各 token 的 logit 梯度为：

$$
\frac{\partial \ell_t^{\text{fKL}}}{\partial z_j}
=
p_j - q_j.
$$

逐一计算：

$$
\begin{aligned}
\frac{\partial \ell_t^{\text{fKL}}}{\partial z_1}
&= 0.2 - 0.7 = -0.5,\\
\frac{\partial \ell_t^{\text{fKL}}}{\partial z_2}
&= 0.3 - 0.2 = +0.1,\\
\frac{\partial \ell_t^{\text{fKL}}}{\partial z_3}
&= 0.5 - 0.1 = +0.4.
\end{aligned}
$$

因此梯度下降会：

- **提高** $t_1$ 的 logit（因为 student 明显低估了 teacher 最支持的 token）；
- **轻微压低** $t_2$ 的 logit（因为 student 略微高估了它）；
- **明显压低** $t_3$ 的 logit（因为 student 严重高估了它）。

把它整理成表格，更直观一些：

| Token | $p_j$ | $q_j$ | rKL 方向 | rKL 力度 | fKL 方向 | fKL 力度 |
|-------|-------|-------|----------|----------|----------|----------|
| $t_1$ | $0.2$ | $0.7$ | ↑ | 强 | ↑ | 强 |
| $t_2$ | $0.3$ | $0.2$ | ↑ **(注意!)** | 弱 | ↓ | 弱 |
| $t_3$ | $0.5$ | $0.1$ | ↓ | 强 | ↓ | 强 |

可以看到：$t_1$ 和 $t_3$ 在两种 KL 下更新方向一致，但 $t_2$ 的更新方向 **完全相反**——这正是 reverse-KL（mode-seeking）与 forward-KL（mode-covering）在优化行为上的核心差异。forward-KL 的行为与直觉高度一致：**哪个 token 概率偏高就压低，哪个 token 概率偏低就抬高。**  
与前面的 reverse-KL 相比，它不会先看“这个 token 的失配程度是否高于全局平均”，而是直接做 **逐 token 的点对点校正**。

**forward-KL 与 reverse-KL 的差异**

虽然两者都属于“完整词表概率对齐”，但它们的优化偏好并不相同：

- **forward-KL** 更关注 teacher 所有的区域是否被 student 覆盖到。因此它倾向于让 student 去 **覆盖 teacher 的所有主要模式**，因此被称为 *mode-covering*。
- **reverse-KL** 更关注 student 当前押注的区域是否得到了 teacher 的认可。因此它倾向于 **打压 student 错押的主峰，并集中到 teacher 更认可的少数模式上**，因此被称为 *mode-seeking*。

#### 小结

所以，SeqKD 与 GKD-style OPD 的共同点在于：它们都属于 **可导的完整词表分布对齐**；  
而它们的关键区别在于：

- SeqKD 是 **teacher rollout / off-policy**，并常采用 **forward-KL**；
- GKD-style OPD 是 **student rollout / on-policy**，并常采用 **reverse-KL**。

### 下一篇预告

PG-style OPD 在梯度更新时会产生和 GKD-style 不一样的走向吗？

---

### 补充：针对 logit 的梯度推导

（以下内容由 GPT5.4 生成）

前文在讨论 GKD-style OPD 的 reverse-KL 与 SeqKD 的 forward-KL 时，直接给出了它们对 student logits 的梯度结论。为了让推导更完整，这里单独证明这两个结果。

设在某个固定前缀 $s_t$ 下，student 的词表大小为 $|\mathcal V|$，其 logits 为

$$
z = (z_1, z_2, \dots, z_{|\mathcal V|}),
$$

softmax 概率为

$$
p_j = \pi_\theta(a_j \mid s_t)
= \frac{e^{z_j}}{\sum_k e^{z_k}},
$$

teacher 分布记为

$$
q_j = \pi_{\text{teacher}}(a_j \mid s_t).
$$

其中：

- $p_j$ 依赖于 student 的 logits $z$；
- $q_j$ 是 teacher 给定的常数，与 student 参数无关。

**softmax 的导数**

后续推导都要用到 softmax 的经典导数公式：

$$
\boxed{
\frac{\partial p_v}{\partial z_j}
=
p_v(\delta_{vj} - p_j)
}
$$

其中 $\delta_{vj}$ 是 Kronecker delta：

$$
\delta_{vj}=
\begin{cases}
1, & v=j,\\
0, & v\neq j.
\end{cases}
$$

这个结果可以由 softmax 直接求导得到：

$$
p_v = \frac{e^{z_v}}{\sum_k e^{z_k}}.
$$

对 $z_j$ 求导，

$$
\frac{\partial p_v}{\partial z_j}
=
\frac{
\delta_{vj} e^{z_v}\sum_k e^{z_k}
-
e^{z_v} e^{z_j}
}{
\left(\sum_k e^{z_k}\right)^2
}
=
\frac{e^{z_v}}{\sum_k e^{z_k}}
\left(
\delta_{vj}
-
\frac{e^{z_j}}{\sum_k e^{z_k}}
\right),
$$

即

$$
\frac{\partial p_v}{\partial z_j}
=
p_v(\delta_{vj} - p_j).
$$

这个式子揭示了 softmax 的耦合性：提高某个 token 的 logit $z_j$，不仅会提高它自己的概率 $p_j$，还会压低其他所有 token 的概率。

#### 1. reverse-KL 对 logit 的梯度推导

单步 reverse-KL 定义为

$$
\ell_t^{\text{rKL}}
=
\mathrm{KL}(p\|q)
=
\sum_v p_v \log \frac{p_v}{q_v}.
$$

我们的目标是计算

$$
\frac{\partial \ell_t^{\text{rKL}}}{\partial z_j}.
$$

**第一步：对 $p_v$ 求偏导**

将 loss 看成 $p$ 的函数：

$$
\ell_t^{\text{rKL}}
=
\sum_v p_v \log \frac{p_v}{q_v}.
$$

对某个 $p_v$ 求偏导：

$$
\frac{\partial \ell_t^{\text{rKL}}}{\partial p_v}
=
\frac{\partial}{\partial p_v}
\left(
p_v \log \frac{p_v}{q_v}
\right)
=
\log \frac{p_v}{q_v} + 1.
$$

**第二步：链式法则**

利用链式法则，

$$
\frac{\partial \ell_t^{\text{rKL}}}{\partial z_j}
=
\sum_v
\frac{\partial \ell_t^{\text{rKL}}}{\partial p_v}
\frac{\partial p_v}{\partial z_j}.
$$

代入前面的两个结果：

$$
\frac{\partial \ell_t^{\text{rKL}}}{\partial z_j}
=
\sum_v
\left(
\log \frac{p_v}{q_v} + 1
\right)
p_v(\delta_{vj} - p_j).
$$

**第三步：展开并化简**

将上式拆成两项：

$$
\frac{\partial \ell_t^{\text{rKL}}}{\partial z_j}
=
\sum_v
\left(
\log \frac{p_v}{q_v} + 1
\right)
p_v\delta_{vj}
-
\sum_v
\left(
\log \frac{p_v}{q_v} + 1
\right)
p_v p_j.
$$

第一项中，$\delta_{vj}$ 只在 $v=j$ 时为 1，因此

$$
\sum_v
\left(
\log \frac{p_v}{q_v} + 1
\right)
p_v\delta_{vj}
=
p_j\left(\log \frac{p_j}{q_j} + 1\right).
$$

第二项中，$p_j$ 与求和变量 $v$ 无关，可以提出来：

$$
\sum_v
\left(
\log \frac{p_v}{q_v} + 1
\right)
p_v p_j
=
p_j
\sum_v
p_v
\left(
\log \frac{p_v}{q_v} + 1
\right).
$$

继续展开：

$$
=
p_j
\left(
\sum_v p_v \log \frac{p_v}{q_v}
+
\sum_v p_v
\right).
$$

由于 $\sum_v p_v = 1$，因此

$$
=
p_j
\left(
\sum_v p_v \log \frac{p_v}{q_v}
+ 1
\right).
$$

于是整体变为

$$
\frac{\partial \ell_t^{\text{rKL}}}{\partial z_j}
=
p_j\left(\log \frac{p_j}{q_j} + 1\right)
-
p_j
\left(
\sum_v p_v \log \frac{p_v}{q_v}
+ 1
\right).
$$

两边的常数项 $+1$ 相消，得到

$$
\boxed{
\frac{\partial \ell_t^{\text{rKL}}}{\partial z_j}
=
p_j
\left(
\log \frac{p_j}{q_j}
-
\sum_v p_v \log \frac{p_v}{q_v}
\right)
}
$$

而 $\sum_v p_v \log \frac{p_v}{q_v} = \mathrm{KL}(p\|q)$，因此也可写成更紧凑的形式：

$$
\boxed{
\frac{\partial \ell_t^{\text{rKL}}}{\partial z_j}
=
p_j
\left(
\log \frac{p_j}{q_j}
-
\mathrm{KL}(p\|q)
\right)
}
$$

**结果解释**

这个式子表明，reverse-KL 对某个 logit 的更新，并不只由该 token 的局部差异 $\log \frac{p_j}{q_j}$ 决定，还会与整个分布的平均失配程度 $\mathrm{KL}(p\|q)$ 比较。因此它体现的是一种 **全局重分配** 行为，而不是简单的逐 token 点对点拉齐。

#### 2. forward-KL 对 logit 的梯度推导

单步 forward-KL 定义为

$$
\ell_t^{\text{fKL}}
=
\mathrm{KL}(q\|p)
=
\sum_v q_v \log \frac{q_v}{p_v}.
$$

由于 $q_v$ 与 student 参数无关，这个式子可以写成

$$
\ell_t^{\text{fKL}}
=
\underbrace{\sum_v q_v \log q_v}_{\text{常数}}
-
\sum_v q_v \log p_v.
$$

因此只需要对 $-\sum_v q_v \log p_v$ 求导。

**第一种推法：直接使用 $\log p_v$ 的导数**

先写出 softmax 的一个常用恒等式：

$$
\log p_v
=
z_v - \log \sum_k e^{z_k}.
$$

因此对 $z_j$ 求导：

$$
\frac{\partial \log p_v}{\partial z_j}
=
\delta_{vj} - p_j.
$$

于是

$$
\frac{\partial \ell_t^{\text{fKL}}}{\partial z_j}
=
-\sum_v q_v \frac{\partial \log p_v}{\partial z_j}
=
-\sum_v q_v(\delta_{vj} - p_j).
$$

展开化简后：

$$
\boxed{
\frac{\partial \ell_t^{\text{fKL}}}{\partial z_j}
=
p_j - q_j
}
$$

**第二种推法：通过链式法则**

对 $p_v$ 求偏导：$\frac{\partial \ell_t^{\text{fKL}}}{\partial p_v} = -\frac{q_v}{p_v}$，再利用 $\frac{\partial p_v}{\partial z_j} = p_v(\delta_{vj} - p_j)$，后续化简与上面完全相同，最终得到同样的 boxed 结果。

**结果解释**

forward-KL 的梯度形式非常直接：$\frac{\partial \ell_t^{\text{fKL}}}{\partial z_j} = p_j - q_j$。因此它是一个典型的 **逐 token 点对点校正**：

- 若 $p_j > q_j$，则梯度为正，梯度下降会压低该 token；
- 若 $p_j < q_j$，则梯度为负，梯度下降会抬高该 token。

这与 reverse-KL 的全局重分配行为形成了鲜明对比：

- **forward-KL** 更像“把 student 的每个概率值逐项拉向 teacher”；
- **reverse-KL** 更像“优先打压 student 错押的主峰，并重塑整个分布形状”。

---

## （4）PG-style 的更新模式

### 前言

前两节我们分别从经典的 Knowledge Distillation 和 RL 的视角介绍了 GKD-style 和 PG-style 的 OPD，第三节我们对其中的 GKD-style 的优化动力学做了更细致的分析，特别是与 SeqKD 相比其 mode-seeking 的优化差异。  
这一节我们介绍 PG-style OPD 是如何对 model 参数进行更新的，并在最后与 GKD-style 做一个正面对比。

### 1）Policy Gradient 的更新

#### Loss 函数

第二节我们简单提到了 PG-style OPD 可简单视为把 RLVR 的奖励信号密集化到每个 token 上，这里我们直接给出 PG-style OPD 的 sequence 级损失函数（注意这里没有采用 RL 的 advantage 写法；在 RL 场景下，我们希望 reward/advantage 越大越好）为：

$$
L_{\text{PG-OPD}}(\theta)
=
\mathbb{E}_{\tau \sim \pi_\theta(\cdot \mid x)}
\left[
\sum_{t=1}^{T} \operatorname{sg}(C_t) \cdot \log \pi_\theta(a_t \mid s_t)
\right]
$$

其中 $a_t$ 是 student 在位置 $t$ 实际采样出的 token，$s_t = (x, a_1, \dots, a_{t-1})$ 是对应的前缀，而 $C_t$ 是 reverse-KL 的单样本近似（Schulman K1 近似）：

$$
C_t = \log \pi_\theta(a_t \mid s_t) - \log \pi_{teacher}(a_t \mid s_t).
$$

注意这里 **只计算了实际采样出来的 token $a_t$** 的 log-ratio，而不是对完整词表分布求和的精确 reverse-KL。  
这里的 $C_t$ 更适合理解为一个单样本的 cost / penalty 信号（不是 reward）。它在单样本上可正可负，但其在 student 分布下的期望为 reverse-KL，因此优化目标并不是让每一步的 $C_t$ 都尽可能小，而是让其期望尽可能趋近于 0（即 student 和 teacher 分布完全一致）。

$\operatorname{sg}(\cdot)$ 表示 stop-gradient：$C_t$ 在反向传播时被当作常数，不对 $\theta$ 产生梯度。

另外一个需要注意的是，PG-style 由于涉及到求期望（解决不可导的问题），而在实际工程中，我们一般通过 Monte Carlo 采样去近似期望。  
具体来说，需要通过采样 $N$ 条轨迹做蒙特卡洛近似（这和所有基于 policy gradient 的 RL 方法是一致的）：

$$
L_{\text{PG-OPD}}(\theta)
\approx
\frac{1}{N}\sum_{i=1}^{N}
\sum_{t=1}^{T^{(i)}} \operatorname{sg}(C_t^{(i)}) \cdot \log \pi_\theta(a_t^{(i)} \mid s_t^{(i)}),
\qquad \tau^{(i)} \sim \pi_\theta(\cdot \mid x)
$$

#### 梯度

下面我们依然从梯度去分析 model 的更新方式。为了方便书写，下面都只考虑采样一次的情形，即 $N=1$（严格来说，即使在单次采样情形下，采样轨迹 $\tau \sim \pi_\theta$ 本身也依赖于参数 $\theta$，所以完整求导会包含对轨迹的求导，这里我们暂时不考虑轨迹的变化，这也是 RL 领域常见做法，感兴趣的朋友可以了解 REINFORCE 风格的 surrogate）。

对 Loss 求导（$C_t$ 被 stop-gradient，视为常数），得到 PG-style OPD 的梯度：

$$
\boxed{
\nabla_\theta L_{\text{PG-OPD}}(\theta)
=
\sum_t \operatorname{sg}(C_t) \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t)
}
$$

注意和上一节 GKD-style 梯度的关键区别：

- GKD-style 的梯度是 $\sum_t \sum_v (\cdots) \nabla_\theta \pi_\theta(a_v \mid s_t)$ ——对 **词表中所有 token $v$** 求和；
- PG-style 的梯度是 $\sum_t (\cdots) \nabla_\theta \log \pi_\theta(a_t \mid s_t)$ ——只涉及 **实际采样出的那一个 token $a_t$**。

这意味着：PG-style 的梯度信息来源只在被采样到的 token 上（当然也会间接影响其他 token 的概率），而 GKD-style 的梯度针对所有 token 产生，后者的信息量天然更大。

和上一节一样，为了直观分析“某个 token 的 logit 会被推高还是压低”，我们仍然把梯度推到 logit 层面。

### 2）从 logit 角度理解 PG-style 的更新

设 student 在位置 $t$ 的 logits 为 $z = (z_1, \dots, z_{|\mathcal{V}|})$，softmax 概率为 $p_v = \pi_\theta(a_v \mid s_t)$，teacher 概率简记为 $q_v = \pi_{teacher}(a_v \mid s_t)$。

PG-style 的 loss 在单步上为

$$
\ell_t^{\text{PG}} = \operatorname{sg}(C_t) \cdot \log p_k,
$$

其中 $k$ 是实际被采样出的那个 token 的索引（即 $a_t = a_k$），$C_t = \log p_k - \log q_k$。

利用上一节推导过的 softmax 求导恒等式

$$
\frac{\partial \log p_k}{\partial z_j} = \delta_{kj} - p_j,
$$

可以得到：

$$
\frac{\partial \ell_t^{\text{PG}}}{\partial z_j}
=
\operatorname{sg}(C_t) \cdot (\delta_{kj} - p_j).
$$

这个式子展开来看就很直观了：

- 对当前 **被采样的 token** $a_k$（即 $j = k$）：

  $$
  \frac{\partial \ell_t^{\text{PG}}}{\partial z_k}
  =
  \operatorname{sg}(C_t) \cdot (1 - p_k);
  $$

- 对当前 **未被采样的 token** $a_j$（即 $j \neq k$）：

  $$
  \frac{\partial \ell_t^{\text{PG}}}{\partial z_j}
  =
  \operatorname{sg}(C_t) \cdot (0 - p_j)
  =
  -\operatorname{sg}(C_t) \cdot p_j.
  $$

注意 $p_k$ 和 $p_j$ 都是大于 0 小于 1 的，那么可以看出，$\frac{\partial \ell_t^{\text{PG}}}{\partial z_k}$ 和 $\frac{\partial \ell_t^{\text{PG}}}{\partial z_j}$ 的正负始终是相反的！  
且梯度下降更新为 $z_j \leftarrow z_j - \eta \frac{\partial \ell_t^{\text{PG}}}{\partial z_j}$，因此：

- 若 $C_t > 0$（即 student 概率高于 teacher，student 过度押注了这个 token）：
  - 被采样 token $a_k$：$\frac{\partial \ell_t^{\text{PG}}}{\partial z_k} > 0$，梯度下降会 **降低** $z_k$ $\Rightarrow$ 减小该 token 的概率；
  - 词表中其他所有 token $a_j$：$\frac{\partial \ell_t^{\text{PG}}}{\partial z_j} < 0$，梯度下降会 **提高** $z_j$ $\Rightarrow$ 无差别地抬高其他所有 token。
- 若 $C_t < 0$（即 student 概率低于 teacher，teacher 更认可这个 token）：
  - 被采样 token $a_k$：$\frac{\partial \ell_t^{\text{PG}}}{\partial z_k} < 0$，梯度下降会 **提高** $z_k$ $\Rightarrow$ 增大该 token 的概率；
  - 词表中其他所有 token $a_j$：$\frac{\partial \ell_t^{\text{PG}}}{\partial z_j} > 0$，梯度下降会 **降低** $z_j$ $\Rightarrow$ 无差别地压低其他所有 token。

这揭示了 PG-style 更新的一个本质特征：**它对未被采样的 token 是完全“盲目”的。**  
当 teacher 不认可 student 采样的 token 时，PG-style 会降低该 token 的概率，但它 **无差别地把概率质量分散给词表中所有其他 token**（当然，这里说无差别可能不太严谨，或者说不区分 teacher 偏好）——不管这些 token 是 teacher 高度认可的，还是同样不认可的。它没有能力区分“应该把概率转移给谁”。

### 3）具体例子

仍然使用上一节的三 token 例子：

$$
p = (0.2,\; 0.3,\; 0.5), \qquad q = (0.7,\; 0.2,\; 0.1).
$$

假设 student 在这个位置采样到了 $t_3$（概率最高的那个 token），则

$$
C_t = \log p_3 - \log q_3 = \log 0.5 - \log 0.1 \approx +1.61.
$$

$C_t > 0$，说明 student 在 $t_3$ 上过度押注了（teacher 只给 $t_3$ 分配了 0.1 的概率，student 却给了 0.5）。

各个 token 的 logit 梯度为：

- **被采样的 $t_3$**：$\frac{\partial \ell_t^{\text{PG}}}{\partial z_3} = (+1.61)(1 - 0.5) = +0.81$。梯度下降会 **降低** $z_3$ $\Rightarrow$ 压低 $t_3$ 的概率。**方向正确。**
- **未被采样的 $t_1$**：$\frac{\partial \ell_t^{\text{PG}}}{\partial z_1} = -(+1.61) \cdot 0.2 = -0.32$。梯度下降会 **提高** $z_1$ $\Rightarrow$ 抬高 $t_1$ 的概率。**方向正确**（teacher 确实最支持 $t_1$），但这 **只是巧合**——PG 并不知道 teacher 对 $t_1$ 的态度，它只是在无差别地抬高所有非采样 token。
- **未被采样的 $t_2$**：$\frac{\partial \ell_t^{\text{PG}}}{\partial z_2} = -(+1.61) \cdot 0.3 = -0.48$。梯度下降会 **提高** $z_2$ $\Rightarrow$ 抬高 $t_2$ 的概率。这里可以发现 **$t_2$ 的更新方向就错了**，因为 PG 在提高除了被采样 token 以外的所有 token 的概率。

### 4）与 GKD-style 的正面对比

现在我们把同一个例子下，GKD-style（reverse-KL）和 PG-style 的 logit 梯度放在一起比较。

回顾上一节的 GKD-style 结果：

$$
\frac{\partial \ell_t^{\text{rKL}}}{\partial z_j}
= p_j \left(\log \frac{p_j}{q_j} - \mathrm{KL}(p\|q)\right),
\qquad \mathrm{KL}(p\|q) \approx 0.68.
$$

整理为表格（**红色**表示与 teacher 对齐方向相反）：

| Token | $p_j$ student | $q_j$ teacher | GKD logit 梯度 | GKD 方向 | PG logit 梯度 | PG 方向 |
|-------|---------------|---------------|----------------|----------|---------------|---------|
| $t_1$ | $0.2$ | $0.7$ | $0.2 \times (-1.93) = -0.39$ | ↑ 提升 | $-0.32$ | ↑ 提升 |
| $t_2$ | $0.3$ | $0.2$ | $0.3 \times (-0.27) = -0.08$ | **↑ 微弱提升** | $-0.48$ | **↑ 较强提升** |
| $t_3$（采样） | $0.5$ | $0.1$ | $0.5 \times (0.93) = +0.47$ | ↓ 压低 | $+0.81$ | ↓ 压低 |

从这张表可以读出两种方法的一些差异：

1. **$t_1$（teacher 最支持的 token）**：两者都抬高。
2. **$t_2$（teacher 不太在意的 token）**：GKD-style 几乎不动它（梯度仅 $-0.08$），因为它知道 teacher 对 $t_2$ 的评价只是中性的；PG-style 却给了较大的抬升力度（$-0.48$），因为它不知道 teacher 对 $t_2$ 的态度，只是在盲目地把从 $t_3$ 拿走的概率然后分配给所有非采样 token。当然，如果是 SeqKD 式的 forward-KL，那么 $t_2$ 概率应该下降以对齐 teacher。
3. **$t_3$（student 错押的主峰）**：两者都压低。

总结来看：

- **GKD-style** 像一位知道答案的老师：它清楚词表中每个 token 应该得到多少概率，因此能 **精准地** 告诉 student “把概率从哪里拿走、分配给谁”。
- **PG-style** 像一位只能打分的考官：它只能告诉 student “你刚才采样的这个 token 好不好”，至于概率应该转移给谁，student 只能 **盲目地** 重新分配。

这也解释了为什么在实践中，PG-style 通常方差更高、收敛更慢（也包括 RL）：每一步更新只利用了 teacher 在 **一个 token** 上的信息，而 GKD-style 同时利用了 teacher 在 **整个词表** 上的信息。

### 5）PG-style 的采样依赖性

上面的分析还揭示了 PG-style 的另一个重要特性：**它的更新结果高度依赖于实际采样到了哪个 token**。

仍然用同一个例子，如果 student 这次碰巧采样（假设非贪婪采样）到的不是 $t_3$，而是 $t_1$，则

$$
C_t = \log p_1 - \log q_1 = \log 0.2 - \log 0.7 \approx -1.25.
$$

$C_t < 0$，teacher 非常认可这个选择。此时的 logit 梯度为：

| Token | $p_j$ | $q_j$ | logit 梯度 | 更新方向 | 力度 |
|-------|-------|-------|------------|----------|------|
| $t_1$（采样） | $0.2$ | $0.7$ | $(-1.25)(1-0.2)=-1.00$ | ↑ 提升 | 强 |
| $t_2$ | $0.3$ | $0.2$ | $-(-1.25)(0.3)=+0.38$ | ↓ 压低 | 中 |
| $t_3$ | $0.5$ | $0.1$ | $-(-1.25)(0.5)=+0.63$ | ↓ 压低 | 强 |

有趣的是，这次 $t_2$ 会被 **压低**（而采样 $t_3$ 时它会被抬高）。至于 $t_2$ 到底应该被抬高还是压低？PG-style **完全无法给出一致的答案**——它的判断完全取决于“这次碰巧采样到了谁”。

而 GKD-style 则完全不受采样影响：不管 student 采样到了哪个 token，GKD-style 在这个前缀上的 logit 梯度永远是确定的（因为它依赖的是完整的 $p$ 和 $q$ 分布，而不是某个 sampled token）。

这就是 PG-style 方差高的 **根本来源**：同一个前缀、同一对分布，仅仅因为采样到不同的 token，梯度更新的方向就可以截然不同。而 GKD-style 在相同条件下的梯度是 **确定性的**。

当然，这不是说 PG-style 优化方向是错的。虽然单样本梯度高度随机，但对采样分布取期望后，它所优化的目标仍然是有确定方向的，只是问题在于单样本估计噪声大。

### 6）小结

| | **GKD-style** | **PG-style** |
|---|---------------|--------------|
| 梯度涉及的 token | 词表中所有 token | 仅采样到的 1 个 token |
| teacher 信息利用 | 完整词表分布 | 仅 sampled token 的 logprob |
| 对未采样 token 的处理 | 精准分配（知道每个 token 该得多少） | 盲目均分 |
| 梯度确定性 | 确定的（给定 $p, q$） | 随采样结果波动 |
| 方差 | 低 | 高 |
| 更像什么 | 监督学习（SFT/KD） | 强化学习（REINFORCE） |

从这个对比可以看出，**若只看蒸馏信号本身，GKD-style 提供了比 PG-style 更细粒度、更低方差的监督**：PG-style 能做到的（压低错误 token、抬高正确 token），GKD-style 都能做到，而且 GKD-style 还能精确控制概率应该转移给谁——这是 PG-style 做不到的。

那 PG-style 还有存在的价值吗？有。它的独特优势不在“信息利用效率”，而在于：

- **与 RLVR 的无缝融合**：PG-style 的 $C_t$ 接口可以直接叠加传统 RL 的标量 reward（如 outcome correctness），这是 GKD-style 难以自然做到的；
- **对采样策略本身的优化**：PG-style 相对来说对 teacher 的依赖较弱，一方面的缺点当然是学习起来可能比较慢，毕竟指导不足；但这也可以说是它的独特之处：充分发挥 student 的主观能动性，让他自己充分探索，我们只关注最终的结果，不关注过程中可能出现的错误或不合理的地方。从这一点来说，PG-style 天然更有利于 student 突破 teacher 的能力——而 GKD-style 说到底还是让 student 尽可能和 teacher 保持一致，从而上限也被 teacher 锁死了。（当然个人认为与纯 RLVR 相比，这两种模式下 student 的更新都是受限的，这也是蒸馏的一个长久命题吧——如何让 student 达到甚至超越 teacher，也已经有不少工作了。）

### 下一篇预告

至此我们已经分别分析了 GKD-style 和 PG-style 的优化动力学。其实也可以看出，OPD 在算法上还有很多可以研究的地方，例如 RLVR 社区里的工作，能否自然应用到 PG-style 上来呢？后面我们再去看看如何对 OPD 做改进。
