---
title: "Sketch on Operational Research"
summary: "运筹学、线性规划、整数规划、近似算法、在线算法与机制设计的课程笔记。"
date: 2026-08-28
lastmod: 2026-08-28
authors:
  - me
tags:
  - 运筹学
  - 最优化
  - 线性规划
  - 算法
toc: false
links:
  - type: site
    url: https://github.com/Floraison0117/personal-website
---

# Sketch on Operational Research

## Intro: Operational Research, Optimization and Linear Programming

运筹学研究如何利用有限资源，使利益最大化或资源消耗最小化；最优化关注建立模型后如何求出最优解。线性规划是优化领域中研究最早、应用广泛且方法成熟的重要分支，也是理解后续整数规划、近似算法和在线算法的基础。

## Linear Programming

### Geometry Foundation

仿射集对任意仿射组合封闭，凸集对任意凸组合封闭。线性方程组的解集是仿射集，而仿射集必为凸集。

超平面、半空间和多面体分别对应线性等式、线性不等式及其有限交集：

\[
H=\{x:a^Tx=b\},\qquad P=\{x:Ax\le b\}.
\]

多面体的顶点、极点和基本可行解是代数结构与几何结构之间的桥梁。回收锥刻画可行域的无界方向：

\[
\operatorname{rec}(P)=\{d:Ad\le 0\}.
\]

分离超平面定理说明，闭凸集外的点可以被一个超平面与集合分隔。Minkowski–Weyl 定理则给出多面体的顶点—射线表示：

\[
P=\operatorname{conv}\{v_1,\ldots,v_k\}+\operatorname{cone}\{u_1,\ldots,u_l\}.
\]

线性规划基本定理指出：若尖多面体上的最大化问题有有限最优值，则最优解可以在某个顶点处取得。

### Simplex Method

通过松弛变量将不等式约束化为标准形式：

\[
Ax\le b\quad\Longrightarrow\quad Ax+x_s=b,\qquad x_s\ge0.
\]

单纯形法将变量分成基变量与非基变量，以基本可行解表示多面体顶点。检验数为

\[
\Delta_j=c_j-c_B^TB^{-1}A_j.
\]

最大化问题中，若所有非基变量的检验数均不大于零，当前基本可行解即为最优解。算法反复选择改进方向、依据最小比值规则确定出基变量，并更新基矩阵。

当初始基本可行解无法直接得到时，可以使用大 M 法或两阶段法引入人工变量。退化可能导致循环，Bland 规则通过选择下标最小的候选变量保证有限步终止。

### Linear Programming Duality

原始问题

\[
\max\{c^Tx:Ax\le b,\ x\ge0\}
\]

对应的对偶问题为

\[
\min\{b^Ty:A^Ty\ge c,\ y\ge0\}.
\]

对偶变量可以解释为资源的影子价格。弱对偶性给出

\[
c^Tx\le b^Ty,
\]

强对偶性则说明在有限最优解存在时，原始和对偶最优值相等：

\[
c^Tx^*=b^Ty^*.
\]

互补松弛条件为

\[
y_i^*(b_i-A_i x^*)=0,\qquad x_j^*((A^T)_jy^*-c_j)=0.
\]

Farkas 引理、Gordan 引理、对偶单纯形法和原始—对偶法进一步揭示了可行性、最优性与证书之间的关系。

## Advanced Linear Programming and Large Scale Decomposition

### Sensitivity Analysis

灵敏度分析研究目标系数、约束右端项以及变量和约束变化对最优基与最优值的影响。影子价格可以表示为最优值对资源上限的边际变化：

\[
y_i^*=\frac{\partial z^*}{\partial b_i}.
\]

### Polynomial-Time Algorithms and Interior-Point Methods

椭球法证明了线性规划的多项式时间可解性。KKT 条件将原始可行性、对偶可行性和互补松弛统一起来。仿射尺度法、障碍函数法、中心路径、势函数下降和 Newton 法构成内点算法的主要工具。

### Large Scale Decomposition

列生成在受限主问题和定价子问题之间迭代；Dantzig–Wolfe 分解按变量结构拆分问题，Benders 分解则按约束结构拆分问题。二者都利用大规模问题中的块结构降低计算复杂度。

## Integer Programming and Combinatorial Optimization

整数规划要求部分或全部变量取整数。线性松弛、全酉模矩阵、分支定界、割平面和动态规划构成精确求解框架。

拟阵以独立系统和秩函数抽象贪心算法的正确性。对于拟阵，按权重从高到低选择仍保持独立性的元素，可以得到最优解；这解释了 Best-In 与 Worst-Out 策略在不同结构下的表现。

## Approximation Algorithm Design and Analysis

近似比衡量算法解与最优解之间的差距，整数间隙则描述线性规划松弛的损失。PTAS、EPTAS、FPTAS 和 APX 刻画不同精度与复杂度之间的平衡。

LP-based approximation 通常遵循“松弛—求解—舍入”流程。典型问题包括背包、顶点覆盖、无容量设施选址和集合覆盖。

图与树上的近似方法包括 Steiner Tree、Metric TSP、最短 Hamilton 路、MST shortcut 和 Christofides 算法。Christofides 算法通过最小生成树、奇度顶点的最小权完美匹配和欧拉回路构造 1.5-近似解。

装箱问题的在线与离线启发式包括 Next Fit、First Fit、Best Fit、Worst Fit 和 First Fit Decreasing。权重函数、调和分组、APTAS、Configuration LP 与 Karmarkar–Karp 算法用于建立更精细的近似界。

## Online Algorithms

在线算法只能根据当前及历史信息作出不可撤销的决策。若对任意请求序列 \(\sigma\) 有

\[
A(\sigma)\le c\,\operatorname{OPT}(\sigma)+\alpha,
\]

则称算法是 \(c\)-竞争的。倍增策略、滑雪租赁、分页与缓存、k-server 和在线装箱问题展示了信息不完备带来的理论代价。

## Game Theory & Mechanism Design

策略型博弈由参与者、策略空间和效用函数组成。纳什均衡要求任何参与者都无法通过单方面偏离提高效用。PoA 与 PoS 分别衡量最坏和最好均衡相对于社会最优的效率损失。

机制设计研究如何在私人信息下设计规则。Vickrey 二价拍卖通过让最高报价者支付第二高报价实现诚实报价；稳定匹配、设施选址博弈、装箱博弈和切蛋糕问题则分别讨论匹配稳定性、资源分摊与公平性。
