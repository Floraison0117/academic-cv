---
title: "Stat-ML 讨论班笔记（一）：统计推断基础"
summary: "涵盖信息论基础（熵、散度、互信息）、指数分布族与广义线性模型、集中不等式。"
date: 2026-08-28
lastmod: 2026-08-28
weight: 3
authors:
  - me
tags:
  - 统计学习
  - 信息论
  - 集中不等式
toc: true
---

# Information Theory Primer

## Shannon Entropy

香农熵衡量随机变量 $X$ 不确定性的大小： $ H(X)=-\sum_x p(x)\log p(x) $

在此基础上，可以定义条件熵 $ H(X | Y=y)=-\sum_x p(x | y)\log p(x | y) $

给定条件会减少不确定性，即 $ H(X | Y)\le H(X) $

## Divergence

散度衡量两个分布的距离，即差异的大小： $ D_f (P || Q) = \sum_x q(x) f\left(\frac{p(x)}{q(x)}\right) $

其中 $f$ 为连续是上凸函数，$f(1)=0$。一个最重要的特例是 KL 散度：$ D_{\mathrm{KL}} (P || Q)=\sum_x p(x) \log \frac{p(x)}{q(x)} $

KL 散度非负。注意到 $P$ 和 $Q$ 不对称，$P$ 的地位是真实分布，$Q$ 的地位是去拟合的分布。KL 散度的一种理解方式是，用分布 $Q$ 来编码 $P$ 时，平均每个符号消耗多少 bit。

Divergence 是比 entropy 更广泛和基础的概念。一方面是因为 KL 散度可以推导出 Shannon entropy。展开 KL 散度：

$$
\begin{aligned}
  D_{\mathrm{KL}}(P || Q) &= \sum_{x \in \mathcal{X}} P(x) [\log P(x) - \log Q(x)] \\
                  &= \sum_{x \in \mathcal{X}} P(x) \log P(x) - \sum_{x \in \mathcal{X}} P(x) \log Q(x) \\
                  &= H(P,Q) - H(P)
\end{aligned}
$$

假设样本空间 $\mathcal{X}$ 的大小为 $n$。若我们选取 $Q$ 为 $\mathcal{X}$ 上的*均匀分布*，即对于任意 $x$，有 $Q(x) = 1/n$。此时

$$ H(P, Q) = - \sum_{x \in \mathcal{X}} P(x) \log (1/n) = - \log (1/n) \sum_{x \in \mathcal{X}} P(x) = \log n $$

于是

$$ H(P) = \log n - D_{\mathrm{KL}}(P || U) $$

由上可知，Shannon entropy $H(P)$ 可以理解为该系统可能达到的最大不确定性 ($\log n$) 减去它当前分布与完全随机分布（均匀分布）之间的距离 ($D_{\mathrm{KL}}$)。

另一方面，KL 散度和互信息可以直接聪离散的 pmf 推广到 pdf，但熵的概念不可以，因为不满足非负性。

其他常用的散度包含 TV, Hellinger 和 $\chi^2$。

## Mutual Information

互信息是两个随机变量的共有信息量： $ I(X,Y)=\sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)} $

互信息是 joint 和 marginal 分布的 KL divergence。直观理解： $ I(X;Y)=H(Y)-H(Y | X) $

## Techniques

*Chain Rule* 令 $X_1, \dots, X_n$ 为离散随机变量，则 $ H(X_1, \dots, X_n)=H(X_1) + H(X_2 | X_1) + \dots + H(X_n | X_1^{n-1}) $

即联合分布是熵可以视为条件熵的逐步叠加。互信息也有类似的链式法则。

*Tensorization of KL and Hellinger Distances* 将 product distribution 的散度用 marginal 之间的散度来表示：

$$ D_{\mathrm{KL}} (P || Q) = D_{\mathrm{KL}} (P_1 \times \dots \times P_n || Q_1 \times \dots \times Q_n)=\sum_{i=1}^n D_{\mathrm{KL}} (P_i || Q_i) $$

类似地，Hellinger distance 也可以方便地 tensorize，但 TV 不行。这解释了为什么要考虑多种 $f$ 散度，因为它们各有擅长。

*Data Processing Inequality* 考虑 Markov chain: $X \to Y \to Z$，必有 $I(X;Z)\le I(X;Y)$，即通信时只可能损失信息而不能增加信息。证明如下：$ I(X;Y,Z)=I(X;Z)+I(X;Y | Z)=I(X;Y) + \underbrace{I(X;Z | Y)}_{=0, \text{Markov 性质}} $

更一般地，定义 Markov chain 的 kernel（不妨理解为转移矩阵）K，必有 $ D_f (K_P || K_Q) \le D_f (P || Q) $

即 Markov chain 不会增加信息。

## Fundamental Limit of Hypothesis Testing

Hypothesis testing 可以理解为根据收集的数据判断哪个关于世界的假设 $P_i$ 是正确的。

*Le Cam's Lemmma* 如果我们只有两个备择假设 $P_1, P_2$，自然界从中随机选择一个。此时错误的概率 $ P( \Phi(X) \neq V)=\frac{1}{2} P_1(\Phi(X)\neq 1) + \frac{1}{2} P_2 (\Phi(X)\neq 2) $

Le Cam 引理给出了错误概率的下界，与 $P_1$ 和 $P_2$ 的距离相关： $ \inf_{\Phi} ( P_1(\Phi(X)\neq 1) + P_2 (\Phi(X)\neq 2)) = 1 - \|P_1 - P_2\|_{\mathrm{TV}} $

其中 $ \|P_1 - P_2\|_{\mathrm{TV}} = 1 - \sup_{A \subset X} (P_1 (A) - P_2 (A)) $

*Pinsker's Inequality* 因为 TV 散度无法 tensorize，我们往往将其放缩为可 tensorize 的散度：$ \|P_1^n - P_2^n\|_{\mathrm{TV}}^2 \le \frac{1}{2} D_{\mathrm{KL}} (P_1^n || P_2^n) = \frac{n}{2} D_{\mathrm{KL}} (P_1 || P_2) $

*Fano's Inequality* 随机变量 $X \in \mathcal{X}$，观察到 $Y$ 后我们希望构造一个估计器 $\hat{X}=f(Y)$，错误概率定义为 $P_e = P(\hat{X} \neq X)$。Fano 给出了 $P_e$ 的下界：$ H(P_e) + P_e \log (|\mathcal{X}|-1) \ge H(X | Y) $

# Exponential Families and Statistical Modeling

## Sufficient Statistic and Exponential Family Models

设 $X$ 是来自于分布 $p_{\theta} (x)$ 的样本。若在给定统计量 $T(X) = t$ 的条件下，$X$ 的条件分布与参数 $\theta$ 无关，则称 $T(X)$ 是 $\theta$ 的 充分统计量。

我们通常使用 Neyman-Fisher 因子分解定理判定充分性。 $T(x)$ 是充分统计量的充要条件为概率密度函数 $ p_{\theta} (x) = h(x) \cdot g_{\theta} (T(x)) $

如果一个分布的 pdf/pmf 可以表示为如下形式，则称其属于指数分布族：

$$ p_{\theta} (x) = h(x) \exp(\langle \theta, \phi(x) \rangle - A(\theta)) $$

其中 $\phi(x)$ 是充分统计量，$h(x)$ 是基础测度，$A(\theta)$ 是 log partition function，用来对 pdf/pmf 实现归一化：

$$ A(\theta) = \log \int_{\mathcal{X}} h(x) \exp(\langle \theta, \phi(x) \rangle) \mathrm{d} x $$

许多常见的分布，比如 Bernoulli 分布，Poisson 分布，正态分布，都属于指数分布族。

指数分布族具有良好的解析性质。比如， $ \nabla A(\theta) = \mathbb{E}_{\theta} [\phi (X)] $

$$ \nabla^2 A(\theta) = \operatorname{Cov}_{\theta} (\phi(X)) \succeq 0 $$

(24) 说明 $A(\theta)$ 是凸函数。如果 $T(x)$ 是 minimal（各参量线性无关），则 $A(\theta)$ 严格凸。

## Fitting an Exponential Family Model

对于任何分布，使用指数分布族去拟合相当于 $ \operatorname{minimize}_{\theta} \ D_{\mathrm{KL}} (P || P_{\theta}) = \int p(x) \log \frac{p(x)}{p_{\theta} (x)} \mathrm{d} x $

这等价于最小化 $ -\int p(x) \log p_{\theta} (x) \mathrm{d} x = -\langle \theta, \mathbb{E}_{P} [\phi (X)] \rangle + A(\theta) $

这说明上述问题关于 $\theta$ 是一个凸优化问题。

换一个视角看，$-\int p(x) \log p_{\theta} (x) \mathrm{d} x=H(P, P_{\theta})$ 是 cross-entropy，所以最小化 KL 散度等价于最小化 cross-entropy。

在实际应用中，我们用来自 $P$ 的 $n$ 个采样 $X_i$ 来代表 $P$。代入上述的式子，最小值在 $ \nabla A(\hat{\theta}_n)=\mathbb{E}_{\hat{\theta}_n}[\phi(X)]=\frac{1}{n} \sum_{i=1}^n \phi(X_i) $

时取到，即匹配了 $\phi(x)$ 的样本矩和一阶矩。这说明最小化 KL 散度等价于极大似然估计。

根据 (27)，我们还可以得到估计参数与实际参数之间的关系： $ \hat{\theta}_n - \theta^* \sim \mathcal{N}(0, n^{-1} \cdot \nabla^2 A(\theta^*)^{-1}) $

这说明 Hessian 矩阵越大，误差越小。

考虑 $\theta$ 的一个局部，得到两个分布 $P_{\theta}$ 和 $P_{\theta + \Delta}$，计算两者之间的 KL 散度可以得到 $ D_{\mathrm{KL}} (P_{\theta} || P_{\theta+\Delta}) = \frac{1}{2} \Delta^T \nabla^2 A(\theta) \Delta + O(\|\Delta\|^3) $

这说明 Hessian 还控制了局部分辨率的大小。

定义 $s(\theta)=\frac{\partial}{\partial \theta} \ln p_{\theta} (x)$ 为 score function，其满足 $\mathbb{E}[s(\theta)]=0$。Fisher 信息量 $I(\theta)$ 被定义为 score 函数的方差： $ I(\theta)=\mathbb{E}\left[ \left(\frac{\partial}{\partial \theta} \ln p_{\theta} (x) \right)^2\right] $

$I(\theta)$ 描述了对数似然函数曲线的弯曲程度。$I(\theta)$ 越大，说明对数似然函数随 $\theta$ 的变化越剧烈，估计越精准。具体而言，Cramer-Rao 下界给出 $ \operatorname{Var}(\hat{\theta})\ge \frac{1}{I(\theta)} $

对于指数分布族，恰有 $I(\theta)=A''(\theta)=\operatorname{Var}(\phi(x))$。

## Generalized Linear Models and Regression

指数分布族可用于解决条件预测问题，即在给定 $X \in \mathcal{X}$ 的情况下预测 $Y \in \mathcal{Y}$： $ p_{\theta} (y | x) = \exp(\phi(x, y)^{\top} \theta - A(\theta | x)) h(y) $

*Linear Regression* 假设：$Y | X = x \sim \mathcal{N}(\theta^{\top} x, \sigma^2)$，则充分统计量 $\phi(x, y) = \frac{1}{\sigma^2} x y$，对数配分函数 $A(\theta | x) = \frac{1}{2 \sigma^2} \theta^{\top} x x^{\top} \theta$。

*Binary Logistic Regression* 假设 $Y \in \{0, 1\}$，则充分统计量 $\phi(x, y) = x y$，对数配分函数 $A(\theta | x) = \log(1 + \exp(x^{\top} \theta))$。

*Multiclass Logistic Regression* 假设 $Y \in \{1, \dots, k\}$，为每个类别分配参数 $\theta_y$。模型形式 $ p_{\theta} (y | x) = \frac{\exp(\theta_y^{\top} x)}{\sum_{j=1}^k \exp(\theta_j^{\top} x)} $

*Poisson Regression* 假设 $Y \in \mathbb{N}$ 充分统计量：$\phi(x, y) = y x$，对数配分函数：$A(\theta | x) = \exp(\theta^{\top} x)$。

统一为 GLM 的好处是可以将这四个问题都看做求解 $ \nabla A(\theta | x)= \mathbb{E}_{\theta} [\phi(X, Y)|X = x] $

这个矩匹配问题。而且，上述四个模型的负对数似然 损失函数都是凸函数，性质良好。

假设 $X$ 服从边缘分布 $Q$，整体联合分布为 $(X, Y) \sim P_{\theta} \circ Q$。通过 Taylor 展开 Bregman 散度，参数微小扰动 $\Delta$ 下的 KL 散度近似为：

$$
  D_{\mathrm{KL}} (P_{\theta} \circ Q || P_{\theta + \Delta} \circ Q) \approx \frac{1}{2} \Delta^{\top} \mathbb{E}_{Q} [\nabla^2 A(\theta | X)] \Delta
$$

# Concentration Inequalities

## Basic Tail Inequalities

Concentration Inequalities 是一类研究随机变量集中程度的不等式。相较于大数定律，它们更精确地描述了有限样本的随机变量偏离其期望值的概率。

*弱大数定律*

$$ \lim_{n \to \infty} P \left( \left|\frac{1}{n} \sum_{i=1}^n X_i - \mu\right| > \epsilon \right) = 0 \quad \text{for any } \epsilon > 0 $$

*强大数定律*

$$ P \left( \lim_{n \to \infty} \frac{1}{n} \sum_{i=1}^n X_i = \mu \right) = 1 $$

基本的 Tail Inequalities 包括 Markov Inequality 和 Chebyshev Inequality：

$$ P(X\ge t)\le \frac{\mathbb{E} [X]}{t}, \quad P(X-\mathbb{E}[X]) \ge t \le \frac{\operatorname{Var}(X)}{t^2} , \quad P(X-\mathbb{E}[X]) \le -t \le \frac{\operatorname{Var}(X)}{t^2} $$

将矩母函数 $\phi_X (\lambda)=\mathbb{E}[e^{\lambda X}]$ 代入，得到 Chernoff bound： $ P(X \ge t) \le \frac{E[e^{\lambda X}]}{e^{\lambda t}}=\phi_X (\lambda) e^{-\lambda t} $

当 $X \sim \mathcal{N}(0, \sigma^2)$ 时，$ \phi_X (\lambda)=\exp\left(\frac{\lambda^2 \sigma^2}{2}\right)$。关注 Gaussian 的尾部分布，我们定义 $X$ 为 $\sigma^2$-sub-Gaussian，若 $ \mathbb{E}(\exp(\lambda (X-\mathbb{E}[X])))\le \exp\left(\frac{\lambda^2 \sigma^2}{2}\right) $

Rademacher 随机变量（即$\{-1,1\}$上的均匀分布）是 $1$-sub-Gaussian。

另一个重要的引理是 Hoeffding's lemma。 当 $X \in [a,b]$ 为有界随机变量时，$ \mathbb{E}[e^{\lambda (X - \mathbb{E}[x])}]\le \exp\left(\frac{\lambda^2 (b-a)^2}{8}\right) $

其证明的两种方法非常典型。第一种方法使用了 *symmetrization* 技巧，即引入一个独立的 Rademacher 随机变量 $\epsilon$。令 $X'$ 为 $X$ 的一个独立副本，则 $ \phi _{X - \mathbb{E}(X)} (\lambda) = \mathbb{E}[\exp(\lambda(X - \mathbb{E}[X']))] \le \mathbb{E}[\exp(\lambda (X - X'))] = \mathbb{E}[\exp(\lambda \epsilon(X - X'))] $

这里的不等式使用了 Jensen 不等式，最后一个等号成立是因为 $\mathbb{E}[X-X']=0$。接下来， $ \mathbb{E}[\exp(\lambda \epsilon(X - X'))] \le \mathbb{E}\left[\exp\left(\frac{\lambda^2 (X-X')^2}{2}\right)\right]\le \exp\left(\frac{\lambda^2 (b-a)^2}{8}\right) $

为了简化 $\mathbb{E}[X]=0$，第二种方法考虑了矩母函数的对数 $ \psi(\lambda)=\log \phi_X (\lambda) = \log \mathbb{E}[e^{\lambda X}] $

如果考虑 $p_{\lambda} (y) = \frac{e^{\lambda y}}{\mathbb{E} [e^{\lambda y} X]} p(y)$，可以发现 $ \psi'(\lambda)=\mathbb{E}[Y_{\lambda}], \quad \phi''_{\lambda} = \operatorname{Var}(Y_{\lambda}) $

由于 $ \operatorname{Var}(B) = \inf_t \mathbb{E}[(B-t)^2] \le \mathbb{E}\left[\left(B-\frac{(a+b)}{2}\right)^2\right] \le \frac{(a-b)^2}{4} $

于是 $\phi'(\lambda)\le \frac{(b-a)^2}{4} \lambda$。结合 $\psi'(0)=0$ 可以得到 $ \log \phi_X (\lambda) = \psi (\lambda) = \int_0^{\lambda} \psi'(u) \mathrm{d} u \le \int_0^{\lambda} \frac{(b-a)^2}{4} u \mathrm{d} u = \frac{(b-a)^2}{8} $

sub-Gaussian 是根据矩母函数定义的，它也反映了随机变量尾部的性质： $ P(X-\mathbb{E}[X] \ge t) \lor P(X - \mathbb{E}[X]\le -t) \le \exp\left(-\frac{t^2}{2 \sigma^2}\right) $

这个性质还可以被 tensorize：对于独立的 $\sigma_i^2$-sub-Gaussian 随机变量 $X_i$，有

$$ \mathbb{E}\left[\exp\left(\lambda \sum_{i=1}^n (X_i - \mathbb{E}[X_i])\right)\right] \le \exp\left(\frac{\lambda^2 \sum_{i=1}^n \sigma_i^2}{2}\right) $$

$$ P\left(\sum_{i=1}^n (X_i - \mathbb{E}[X_i]) \ge t\right) \lor P\left(\sum_{i=1}^n (X_i - \mathbb{E}[X_i]) \le -t\right) \le \exp\left(-\frac{t^2}{2 \sum_{i=1}^n \sigma_i^2}\right) $$

将矩母函数的分布限制在一个区域内，可以得到 sub-exponential 的概念。一个参数为 $(\tau^2,b)$ 的 sub-exponential 分布是对所有 $|\lambda|\le 1/b$ 均有 $ \mathbb{E}[e^{\lambda(X-\mathbb{E}[X])}] \le \exp\left(\frac{\lambda^2 \tau^2}{2}\right) $

sub-Gaussian 是 b=0 的情形。在 sub-exponential 的背景下，我们可以得到更紧的不等式。比如，当 $X \in [-b,b]$ （这暗示 $\sigma^2\le b^2$）时， $ \mathbb{E}(\exp(\lambda X))\le \exp\left(\frac{3\lambda^2 \sigma^2}{5}\right), \text{ for } |\lambda|\le \frac{1}{2b} $

进一步，如果 $\lambda\ge 0, X\le b, \mathbb{E}[X]=0$ 时，不需要给出 $X$ 的下界，Bernnett's Inequality 有 $ \mathbb{E}(\exp(\lambda X))\le 1 + \frac{\sigma^2}{b^2} (e^{\lambda b} - 1- \lambda b) $

从 sub-exponential 的矩母函数中我们可以得到尾部特征：对于 $\mathbb{E}[X]=0$ 的 $(\tau^2,b)$-sub-exponential， 有 $ P(X\ge t) \lor P(X \le -t) \le \exp\left(-\frac{1}{2} \min \left\{\frac{t^2}{\tau^2}, \frac{t}{b}\right\}\right) $

同样也可以张量化：对于独立的 $(\tau_i^2, b_i)$-sub-exponential 随机变量 $X_i$，有 $ \mathbb{E}\left[\exp\left(\lambda \sum_{i=1}^n a_i X_i\right)\right] \le \exp\left(\frac{\lambda^2 \sum_{i=1}^n a_i^2 \tau_i^2}{2}\right), \text{ for } |\lambda|\le \frac{1}{2 \max_i b_i|a_i|} $

在众多 sub-exponential 的例子中， Bernstein condition 是一个重要的特例。称 $X$ 满足 $b$-Bernstein condition，如果 $ |E[(X-\mu)^k]|\le \frac{k!}{2} \sigma^2 b^{k-2}, \text{ for } k=3,4,\dots $

下述引理指出满足 $b$-Bernstein condition 的随机变量是 $(\sigma^2, b)$-sub-exponential 的：$ \mathbb{E}[e^{\lambda (X-\mathbb{E}[X])}]\le \exp\left(\frac{\lambda^2 \sigma^2}{2(1-b|\lambda|)}\right), \text{ for } |\lambda|\le 1/b $

Bernnett's Inequality 是 Bernstein condition 的一个直接推论。令 $X_i$ 为满足 $\mathbb{E}[X_i]=0, \operatorname{Var}(X_i)=\sigma_i^2, X_i\le b$ 的随机变量。令 $h(t)=(1+t)\log(1+t)-t, \sigma^2=\sum \sigma_i^2$，则 $ P\left(\sum_{i=1}^n X_i \ge t\right) \le \exp\left(-\frac{\sigma^2}{b^2} h\left(\frac{b t}{\sigma^2}\right) \right) $

## Martingale Methods

令 $M_1, M_2, \dots$ 为实值随机变量，如果存在 $\{Z_1, Z_2, \dots\} \in \mathcal{Z}$ 和一列函数 $f_n: \mathcal{Z}^n \to \mathbb{R}$ 使得 $ E[M_n | Z_1^{n-1}] = M_{n-1} \text{ and } M_n = f_n (Z_1^n) $

则 $M_n$ 为鞅序列。我们称 $M_n$ 适应 $\{Z_n\}$。

序列 $D_n$ 被称为鞅差序列，若 $M_n := \sum_{i=1}^n D_i$ 是鞅序列。不难得到，存在 $Z_n, g_n$ 满足 $ E[D_n | Z_1^{n-1}] = 0 \text{ and } D_n = g_n (Z_1^n) $

鞅序列的一个实例是公平赌博。$M_n$ 代表玩到第 $n$ 轮时的财富，$Z_1, \dots, Z_{n-1}$ 是之前观察到的结果，条件 $ E[M_n | Z_1^{n-1}] = M_{n-1}$ 表明，在知道全部的历史信息时，下一轮的财富期望仍然是当前的财富。

另一个重要的鞅差序列是 Doob martingales。令 $f: \mathcal{X} \to \mathbb{R}$ 为任意函数，差分序列 $ D_i := \mathbb{E}[f(X_1^n)| X_1^i] - E[f(x_1^n)|X_1^{i-1}] $

证明使用全期望公式，$ E[D_i | X^{i-1}_1] = E[E[f (X_1^n ) | X_1^i] | X^{i-1}_1 ] - E[f (X_1^n ) | X^{i-1}_1 ] = 0 $

注意到 $M_n = f(X_1^n) - E[f(X_1^n)]$，所以 Doob martingale 刻画了 $f$ 和其期望之间的误差。

称 $D_n$ 为 $\sigma_n^2$-sub-Gaussian martingale difference 序列，若$ E[\exp(\lambda D_n) | Z_1^{n-1}] \le \exp\left(\frac{\lambda^2 \sigma_n^2}{2}\right) $

Azuma-Hoeffding 定理给出，对于 $\sigma_n^2$-sub-Gaussian martingale difference 序列 $\{D_n\}$，$M_n$ 是 $\sum_{i=1}^n \sigma_i^2$-sub-Gaussian，并且 $ \max\{P(M_n \ge t), P(M_n \le -t)\} \le \exp\left(-\frac{t^2}{2 \sum_{i=1}^n \sigma_i^2}\right) $

对应 (69)，可以发现独立的 sub-Gaussian 和 sub-Gaussian martingale 具有类似的性质。事实上，独立是一种特殊的鞅情形。

作为上述性质的直接推论，如果 $|D_i|\le c$ 恒成立（暗示 $\sigma_i^2\le c^2$），则 $ P(n^{-1/2} M_n \ge t) \lor P(n^{-1/2} M_n \le -t) \le \exp\left(-\frac{t^2}{2 c^2}\right) $

这说明一个每步不超过 $c$ 的鞅随机游走偏离期望的程度为 $O(\sqrt{n})$。

## Bounded Differences Inequality

Bounded difference 指 $f$ 单个自变量的改变对函数值的改变可控：$ | f(x_1^{i-1},x_i,x_{i+1}^n) - f(x_1(i-1), x_i', x_{i+1}^n) | \le c_i $

满足上述性质的 $f$ 满足著名的 bounded diferences inequality:

$$
  P (f(X_1^n) - E[f(X_1^n)] \ge t) \lor P (f(X_1^n) - E[f(X_1^n)] \le -t) \le \exp\left(- \frac{2 t^2}{\sum_{i=1}^n c_i^2}\right)
$$

证明思路是使用 Doob martingale 和 Azuma-Hoeffding inequality。

在 Banach 空间中，令 $X_i$ 满足 $E[X_i]=0, \|X_i\|\le c$。由三角不等式，$f(X_1^n):=\left\|\frac{1}{n} \sum X_i\right\|$ 满足 bounded difference，因而若 $X_i$ 独立，有 $ P\left(\left\|\frac{1}{n} \sum_{i=1}^n X_i\right\| - E\left(\left\|\frac{1}{n} \sum_{i=1}^n X_i\right\|\right)\ge t\right) \le 2 \exp\left(-\frac{n t^2}{2 c^2}\right) $

Banach 空间（完备的赋范向量空间）可用 $L^p (1\le p\le \infty)$ 范数理解，Hilbert 空间（完备的内积空间）可用 $L^2$ 范数理解。

另一个例子是 Rademacher complexities，用来衡量一个函数类的表达能力。直观理解，其衡量一个函数类在多大程度上能够拟合纯噪声。定义 empricial Rademacher complexity of $\mathcal{F}$ 为 $ R_n (\mathcal{F} | x_1^n) = \frac{1}{n} E\left[\sup_{f \in \mathcal{F}}\sum_{i=1}^n \epsilon_i f(x_i)\right] $

$\mathcal{F}$ 的 Rademacher complexity 是 $ R_n (\mathcal{F}) = E[R_n (\mathcal{F} | X_1^n)] $

如果 $f: \mathcal{X} \to [b_0, b_1]$，则 Rademacher complexities 满足 bounded difference，由此可知 $R_n (\mathcal{F} | X_1^n)$ 是 $\frac{(b_1-b_0)^2}{4n}$-sub-Gaussian 的。

还可以把 Rademacher complexity 视为一个随机向量。定义 $ P_n^0 = \frac{1}{n} \sum_{i=1}^n \epsilon_i \mathbf{1}_{X_i}, f \to P_n^0 f = \frac{1}{n} \sum_{i=1}^n \epsilon_i f(X_i) $

可以定义

$$
\begin{aligned}
 \|P_n^0\|_{\mathcal{F}}&= \sup_{f \in \mathcal{F}} P_n^0 f = \sup \frac{1}{n} \sum_{i=1}^n \epsilon_i f(X_i) \\
 \mathbb{E}_{\epsilon} [\|P_n^0\|_{\mathcal{F}}]&=R_n (\mathcal{F} | X_1^n)
\end{aligned}
$$
