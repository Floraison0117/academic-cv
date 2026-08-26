---
title: "Stanford CS234：Reinforcement Learning 课程简介与笔记"
summary: "漫谈从 MDP、policy gradient，到 imitation learning、multi-arm bandit、MCTS 和 AlphaZero 。"
date: 2026-08-26

image:
  caption: ''

cover:
  image: "featured-zine-wide.png"
  position:
    x: 50
    y: 40
  overlay:
    enabled: true
    type: "gradient"
    opacity: 0.4
    gradient: "bottom"
  fade:
    enabled: true
    height: "80px"
authors:
  - me

tags:
  - RL
---

和西湖大学赵世钰老师的《强化学习的数学原理》相比，Stanford CS234 覆盖的内容更广。课程不仅涉及 PPO、DPO 等比较先进的 RL 算法，还介绍了 multi-armed bandit 这个 problem setting 下的一系列算法，涉及不少理论分析。本人没有看过课程录像，仅根据 Winter 2026 的 slides 过了一遍。slides 中每介绍完一块新内容，就会列出几个问题以供自查理解，这点不错。课程整体的质量也符合 Stanford 的招牌，包含必要的古典内容和前沿进展。

用四个词概括 RL，或许是：**Generalization**, **Optimization**, **Delayed outcome**, **Exploration**。以下是笔者不成体系的一己之见，碍于水平比较零散。

## Making Sequences of Good Decisions Given a Model of the World

强化学习通过和环境交互进行训练。交互过程可建模为 **Markov Process**：一个 entity 处在某个 state，依据某个 policy 做出某个 action，得到某个 reward。如果 model 已知（即我们知道这个世界的 dynamics 和 reward），在有限的 action space 和 state space 中，通过 **Bellman Equation**，我们可以使用 **policy iteration** 或 **value iteration** 求出最优 policy。

- **Policy evaluation without known dynamics & reward models** 我们可以通过 empirical 的方法了解世界，即通过 Monte Carlo 采样 rollout 来构建 model。利用数据的不同方式可以导出 first-visit MC、every-visit MC 和 incremental MC。Temporal difference 是另一个伟大的发明，它重新定义了 value function 更新的目标 TD target，通过 bootstrap 和 sample 获得比 MC 更低的方差。由此又有 on-policy 和 off-policy 两种范式。
- **Evaluation of the quality of a policy estimation approach** 可以考虑 consistency、computational complexity、memory requirements、statistical efficiency 和 empirical accuracy。
- **Policy improvement** 为了平衡 exploration 和 exploitation，可以强行引入随机性，即 epsilon-greedy（和 P2P 中的 opportunistic sharing 机制很像）。这种简单的数学结构可以方便地分析理论性质。
- **From tabular settings to function approximation** 有限的状态和行为可以列举，但当空间过大时，memory 和 efficiency 都会成为问题。既然要拟合某个未知的东西，那就引入神经网络来表示 policy 和 value，甚至 reward。于是可以引入优化领域的一系列操作，比如 policy gradient、Q-learning、EMA 等。其中有很多工程 trick，比如 baseline、experience replay 和 fixed target，用来提高训练的稳定性。
- **Advanced policy gradient methods** 梯度下降在参数空间中做，但值域空间中的变化是否剧烈，我们不得而知。PPO 引入 gradient clip，限制每一次变化的幅度，减少 performance collapse。也可以加入 adaptive KL penalty 来限制变化幅度。除此之外，还有一个 trick 叫做 GAE，可以在获取更多新数据之前做好几步 gradient steps，提高 data efficiency。

## Imitation Learning

Input：without reward function。

- **Behavior cloning、Inverse RL、Apprenticeship learning via Inverse RL** 能否直接使用监督学习学到专家的 policy？能否自己构建出 reward？能否用 reward 生成一个好的 policy？
- **Compounding errors** BC 的问题是 compounding errors，因为 model 只见过好数据，不知道怎么挽救失败的情形。DAGGER 的做法是记录下失败的情形，并问 teacher 应该怎么办，不过 teacher 也不一定知道。
- **Feature-based reward function** 要学习一个 reward function，那 reward 应该具有怎样的形式和性质呢？人们往往用 feature 的线性组合表示 reward，并在多个可能的 reward function 中选择熵最大的，因为这个 reward 对特定 demonstration 的 additional preference 最小。
- **Learning from human preference** human effort 处在 demonstrations only 和 constant teaching 之间。人们直接评价的噪声较大，但比较的噪声较小。我们可以采用 Bradley-Terry 先验，建模人们认为 option i > option j 的概率，以此做 preference learning。另一种方法是 direct preference optimization（DPO），用到一些恒等变换。

## Multi-Armed Bandits

已知 m 个 actions（arms），reward 是未知的概率分布。在每个时间 t，agent 选择动作 (a_t)，环境返回 (r_t)，目标是最大化 reward 的总和。和 Markov process 的区别是，没有状态转移，状态不影响未来状态，是单步反馈而不是序贯决策。考虑到问题本身具有复杂性，我们往往考虑 regret 而非 reward 本身。Bandits are a simpler place to see the ideas, but these ideas will extend to MDPs。

- **Epsilon-greedy methods** 会得到 linear regret。实际上，explore forever 和 explore never 都会得到 linear total regret，能否进一步得到 sublinear regret 呢？Regret 的 lower bound 分为 problem-independent 和 problem-dependent，后者往往和 gap Δ_a 以及不同 arm 的 distribution 的 KL 散度有关。
- **Optimism under uncertainty** 选择那些可能具有高价值的动作，不仅考虑当前动作的价值，还考虑这个动作的潜力。这样做的结果是，要么得到 high reward，要么 learn something，不会两头落空。Upper confidence bound（UCB）对 sub-Gaussian 随机变量给出潜力的乐观上界，然后采取 greedy 策略。
- **Bayesian regret** 给 reward 一个先验（prior over the unknown parameters），然后通过观测更新先验分布的参数。比如对于 Bernoulli 分布，可以假设其先验为 Beta 分布。融入到 RL 中，有 PSRL（NeurIPS 2013，ICML 2018）。
- **Probability matching / Thompson sampling** 根据某个动作是最优动作的概率来选取动作。Thompson sampling 是 probability matching 的经典实现：先根据已有数据构造未知参数或价值函数的后验分布，然后从后验中采样一个可能的模型或参数，再选择在这个采样模型下最优的动作。这种方式在 contextual multi-armed bandits 问题中表现得很好。
- **Probably approximately correct（PAC）** 以至少 (1-delta) 的概率选择一个 (epsilon)-optimal action，界是关于问题超参数的多项式。第一次听到还是在 Stat-ML 里。经典的算法比如 MBIE-EB。

## Monte Carlo Tree Search

MCTS 的核心思想是：不必先求出整个 state space 上的最优 policy，而是在当前真实状态下，用额外的局部计算来做一个更好的当前动作选择。它从当前状态作为 root 建立搜索树，通过多次模拟、扩展和回传，不断改进对当前动作的价值估计，最后选择搜索树中最有希望的动作。

- **Simple Monte Carlo search** 对每个候选 action，从当前状态出发模拟很多条 trajectory，然后用平均 return 估计这个 action 的好坏，最后选择平均 return 最高的 action。它可以看成只对当前状态做一步 policy improvement；缺点是 simulation policy 通常固定，而且没有充分复用不同模拟之间的树结构信息。
- **Expectimax tree** 如果有完整的 MDP model，可以从当前状态向前展开搜索树，系统地考虑 action 和随机 next state，从而更精确地评估当前动作。但是 expectimax tree 的规模会随 horizon 指数增长，所以在大状态空间、长 horizon 或复杂游戏中很快不可行。
- **Monte Carlo tree search** 用 sampling 替代完整展开。它不是穷举所有可能分支，而是反复从 root 出发进行模拟，每次只访问搜索树中的一部分路径，然后根据模拟结果更新树上的统计量。这样可以把计算资源集中在更有希望的分支上，是一种 selective best-first search。
- **基本流程** 包括 selection、expansion、simulation/evaluation 和 backup，分别负责选择路径、加入新节点、估计新节点价值，以及反向更新祖先节点。
- **Upper Confidence Tree（UCT）search** UCT 是 MCTS 的经典版本，借用 multi-armed bandit 中的 UCB 思想，把搜索树中每一个需要选择动作的节点都看成一个 bandit problem。它同时考虑当前平均表现和探索不足程度，从而平衡 **exploitation** 和 **exploration**。
- **适用场景** 通常是状态空间很大、动作空间相对较小、horizon 较长的问题。它适合在当前状态附近做高质量局部规划；如果动作空间特别大，branching factor 仍然会成为瓶颈。

## AlphaGo / AlphaZero and Neural-Network-Guided MCTS

AlphaGo / AlphaZero 的核心思想是把 neural network 的泛化能力和 MCTS 的局部搜索能力结合起来。围棋这样的游戏规则明确，但状态空间和搜索空间巨大，传统 brute-force game-tree search 很难成功。

- **Policy-value neural network** 给搜索提供 prior 和 value estimate。对一个棋盘状态，它同时输出 policy prior（网络认为哪些动作更可能好）和 value estimate（网络估计当前局面对当前玩家有多有利）；前者指导搜索优先探索哪些动作，后者评估叶子节点，避免每次都必须模拟到终局。
- **PUCT selection** AlphaZero 的搜索受到 UCT 启发，使用 PUCT。它同时考虑搜索树中已有的平均价值、神经网络给出的动作先验，以及该动作目前被访问的次数，因此会优先探索表现好或有潜力但搜索次数还不够多的动作。
- **Repeat many times** 对同一个当前局面，AlphaZero 会重复很多次 selection、expansion/evaluation 和 backup。重复次数越多，根节点下各个动作的访问次数越能反映搜索后的偏好；最后实际落子时，根据 MCTS 搜索后的根节点访问次数形成最终策略。
- **Self-play** AlphaZero 通过 self-play 生成训练数据。每一步都先运行 MCTS 得到更强的 root policy，然后根据这个 policy 选择动作，直到游戏结束并得到胜负结果。这样一局棋可以产生很多训练样本，而且对手总是和自己水平接近，训练过程自然形成一种 curriculum learning。
- **关键 components** self-play、strategic computation、selective best-first search、power of averaging、local computation，以及不断学习和更新 heuristics。
- 它不是纯 model-free RL，也不是纯 tree search，而是把 learned heuristics 和 online planning 合在一起。
