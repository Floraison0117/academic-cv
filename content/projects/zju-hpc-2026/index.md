---
title: "ZJU HPC 2026：实验个人解答"
summary: "简单集群搭建，MoE 向量化计算，GDN Prefill 前向优化，昇腾算子开发与优化，AMSS-NCKU 数值相对论程序优化，INT8 张量核模拟 FP64 GEMM，Gemma4 端到端推理优化，鲲鹏 trsm 优化。"
date: 2026-08-27T12:00:00+08:00
lastmod: 2026-08-27T12:00:00+08:00
math: true
authors:
  - me
links:
  - type: code
    url: https://github.com/Floraison0117/zju-hpc-2026
tags:
  - HPC
---

好一场 Agent 大战！

<!--more-->

以下按 Lab 顺序记录每个实验的主要任务、原理与解法（实验原文档见 [hpc101.zjusct.io](https://hpc101.zjusct.io/lab/)）。

## Lab 1：简单集群搭建

**任务**：在 WSL2 中用 Docker 模拟 node01–node04 四节点 MiniCluster，从源码配置 OpenMPI、BLAS、HPL 软件栈，部署 NFS 共享文件系统与 Slurm 作业调度，最后用 HPL 基准测试评估集群浮点性能。

**原理**：本实验的核心是把一个分布式内存系统拆成四个相互协作的层次。MPI 负责进程间消息传递，NFS 提供统一的程序与数据路径，MUNGE 为 Slurm 节点认证，Slurm 则负责把作业分配给可用节点。HPL 最终求解稠密线性方程组

$$
A X = B,
$$

其中 \(A \in \mathbb{R}^{N \times N}\)。其主要计算是带部分选主元的 LU 分解，浮点操作量近似为

$$
F(N) \approx \frac{2}{3}N^3,
\qquad
\mathrm{GFLOPS}=\frac{F(N)}{T\times 10^9}.
$$

HPL 将矩阵按二维 block-cyclic 方式分布到 \(P\times Q\) 的 MPI 进程网格；\(P Q\) 必须等于 MPI rank 总数。面板分解、行交换和广播构成通信与同步开销，因此增加 rank 后加速不会线性增长。块大小 \(NB\) 同时影响 BLAS Level-3 矩阵乘的缓存复用和面板通信频率：过小会增加通信，过大则可能降低局部缓存效率。

NFS 的作用不是参与数值计算，而是让所有节点看到相同的 `/cluster/shared` 路径；Slurm 分配资源后由 `srun` 启动 MPI rank，rank 再通过 OpenMPI 协作完成 HPL。

**解法**：

- 源码编译 OpenMPI 4.1.6、参考 BLAS/CBLAS 与 HPL 2.3，静态链接进 `xhpl`；`/etc/hosts` 静态解析加 SSH 免密；
- node01 导出 `/cluster/shared`，计算节点以 `nolock` 挂载 NFS，跨节点读写一致；
- 全节点统一 `munge.key`，`slurm.conf` 定义 debug 分区，`sbatch` 加 `srun --mpi=pmix` 启动 HPL；
- 参数扫描：N=4000、NB=64、P×Q=4×2、8 rank 加 hwthread 绑定，22.03 GFLOPS；
- Bonus：更换 BLIS 数学库快 2.88x（59.05 GFLOPS）；另部署 CephFS 与 K3s 验证分布式存储与容器编排。

## Lab 2：MoE 的向量化计算

**任务**：在 Xeon Gold 5418Y 上实现并优化一个 DeepSeek-V3 风格的量化 MoE 层前向计算。并在 RISC-V 平台上实现 MoE 算子。

**原理**：对 token \(x_t\)，Router 先计算专家亲和度并选择 Top-\(K\) 专家：

$$
z_{t,e}=r_e^\top x_t,
\qquad
s_{t,e}=\sigma(z_{t,e})=\frac{1}{1+e^{-z_{t,e}}}.
$$

Top-\(K\) 的选择可以使用带专家偏置的 \(s_{t,e}+b_e\)，但最终混合权重仍由未加偏置的亲和度归一化：

$$
g_{t,e}=\frac{s_{t,e}}{\sum_{j\in\mathcal{S}_t}s_{t,j}},
\qquad |\mathcal{S}_t|=K.
$$

每个路由专家是一个 SwiGLU 前馈网络：

$$
\mathrm{FFN}_e(x)=W_d^{(e)}\left(\mathrm{SiLU}(W_g^{(e)}x)\odot W_u^{(e)}x\right),
$$

因此带共享专家的最终输出为

$$
y_t=x_t+\mathrm{FFN}_{\mathrm{shared}}(x_t)+
\sum_{e\in\mathcal{S}_t}g_{t,e}\mathrm{FFN}_e(x_t).
$$

W8A8 中，激活按 token 求尺度并量化为 INT8：

$$
s_{x,t}=\frac{\max_i|x_{t,i}|}{127},
\qquad
x_{q,t}=\operatorname{round}(x_t/s_{x,t}).
$$

整数点积在 INT32 中累加，再乘 \(s_Ws_{x,t}\) 反量化；SwiGLU 得到的 hidden 还要重新求 scale，才能执行 down 投影。Baseline 以 token 为外层循环，同一专家权重会被不同 token 重复读取，使矩阵乘退化为矩阵向量乘。以 S3 为例，专家权重总量约为 \(3DH\)，参考实现的权重流量和 MAC 数近似为

$$
B_{\mathrm{base}}=N(K+1)\cdot 3DH,
\qquad
C_{\mathrm{base}}=N(K+1)\cdot 3DH,
$$

算术强度约为 \(1\ \mathrm{MAC/byte}\)。按 expert 分组后，每个专家权重只需加载一次：

$$
B_{\mathrm{group}}=(E+1)\cdot 3DH,
\qquad
I_{\mathrm{group}}=\frac{C_{\mathrm{base}}}{B_{\mathrm{group}}}
\approx 37.65\ \mathrm{MAC/byte}.
$$

这解释了为什么 counting sort 和按专家处理 token 能显著加速多 token 场景：它们不改变数学结果，却把权重访问从反复穿过 LLC/L2 变成同一专家在缓存中的复用。单 token 场景则缺乏这种复用，更适合使用 VNNI；分组后的 token 小批次再交给 AMX-INT8 矩阵乘。

**解法**：

- 单 token 路径：AVX-512 VNNI `vpdpbusd`，激活平移 +128 转无符号、预计算权重行和修正，OB=4 让工作集驻留 L1；
- 多 token 路径：先完成 Router、Top-K 与量化，再 counting sort 按专家分组，同专家权重进 L2 复用（S3 即 2.68x）；
- 分组后接 AMX-INT8 批量矩阵乘：`preprocess` 预打包 VNNI 交错布局，按专家组 token 数 M 分派（M≤4 走 VNNI，M≥5 走 AMX）；
- 端到端：Router 按 8 token 分块复用、一次扫描维护 Top-K、持久工作区免 malloc、量化与累加全部 AVX-512 化；
- 线程自适应（S3 4 线程、S4 8 线程），AMX tile 配置按线程局部化；
- Bonus：迁移 RISC-V，RVV FMA 加速 Router，SpaceMiT IME tile 做专家矩阵乘；

## Lab 3：GDN Prefill 前向优化

**任务**：用 TileLang 在 H800 MIG 上实现 GDN（Gated DeltaNet）的 prefill 前向 kernel。

**原理**：GDN 将门控遗忘与 delta update 合并到状态矩阵 \(S_t\) 的递推中。单 token 形式为

$$
\bar S_t=\alpha_tS_{t-1},
\qquad
S_t=\bar S_t+\beta_t k_t^\top(v_t-k_t\bar S_t),
\qquad
o_t=q_tS_t.
$$

这种递推沿时间维存在依赖链，直接逐 token 执行会限制 GPU 并行度；若一次性计算 \(QK^\top\)，又会产生 \(O(L^2)\) 的中间矩阵。Chunk-wise 方法把序列切成 \(C=64\) 的块，在块内用矩阵乘，在块之间只传递状态。令 \(B=\operatorname{Diag}(\beta)\)、\(\Gamma=\operatorname{Diag}(\gamma)\)，其中门控前缀和在对数域计算，\(\gamma_{c,r}=\exp(g^{\mathrm{cumsum}}_{c,r})\)。对第 \(c\) 个 chunk，预计算的下三角逆矩阵 \(A\) 与中间量为

$$
A=\left(I+\operatorname{StrictLower}(B\Gamma KK^\top\Gamma^{-1})\right)^{-1},
$$

$$
U=ABV,
\qquad
W=AB\Gamma K,
$$

$$
S_{c+1}=\gamma_cS_c+\gamma_cK^\top\Gamma^{-1}(U-WS_c),
$$

$$
O_c=\frac{1}{\sqrt{d_k}}
\left[\Gamma QS_c+\Gamma\operatorname{Lower}(QK^\top)\Gamma^{-1}(U-WS_c)\right].
$$

输出必须使用更新前的 \(S_c\)，所以实现时要先保存状态副本，再写入 \(S_{c+1}\)。原始 TileLang baseline 将 \(W\)、\(U\)、\(V_{\mathrm{new}}=U-WS_c\) 和 chunk 内 scores 分别写入 FP32 全局内存，并用五个 kernel 串行完成；主要代价不是理论 FLOP，而是中间张量的全局读写、kernel launch 和 scores 的标量点积。

**解法**：

- 等价数学变换：\(V_{new} = AB(V − ΓKS_c)\)，结合律省一次矩阵乘，再按 A 的下三角结构裁剪无效乘加
- Kernel 融合：全序列 raw \(QK^T\) 一次加每 chunk fused kernel，state 驻留 shared memory
- Persistent kernel：一个 CTA 处理全部 chunk，launch 数从 5N 降到 2，A 用 cp.async 双缓冲预取
- Tensor Core：五次标量归约改为 `T.gemm`（几何平均 14.89x），\(QK^T\) 融合进 persistent kernel、K 只加载一次
- 无条件加载去掉 if 分支使编译器向量化（excessive sectors 47%→14%），Q/K 用 async_copy 做异步流水线
- VALUE_TILE 从 16 增大到 64/128/32 自适应派发，并去掉 FP32 中间缓冲（合计 2.02x）

## Lab 3.5：昇腾算子开发与优化

**任务**：在昇腾 910B4 NPU 上用 Ascend C 实现并优化 FusedAddRmsNorm 融合算子。

**原理**：FusedAddRmsNorm 对每个 batch 行 \(b\) 先做残差相加，再计算 RMS 归一化：

$$
R_b=x_b+\mathrm{residual}_b,
\qquad
\mathrm{rms}_b=\sqrt{\frac{1}{H}\sum_{i=0}^{H-1}R_{b,i}^2+\varepsilon},
$$

$$
y_b=\frac{R_b}{\mathrm{rms}_b}\odot w,
\qquad
\mathrm{residual\_out}_b=R_b,
\qquad \varepsilon=10^{-6}.
$$

该算子同时读写四个 \(B\times H\) 张量，FP16 评测形状下每个元素大约需要 5 次基础浮点操作；以 \(B=1,H=1024\) 为例，读写量约 2 MiB、计算量约 1.31 MFLOP，算术强度约为

$$
I\approx\frac{1.31\ \mathrm{MFLOP}}{2\ \mathrm{MiB}}
\approx0.63\ \mathrm{FLOP/byte},
$$

因此它更接近访存与指令发射受限的逐元素算子，而不是矩阵计算受限。昇腾 910B 的 AIV 负责向量与元素级运算，MTE2 负责 GM→UB、MTE3 负责 UB→GM，Vector 单元只能处理 UB 中的数据。三条流水线本可重叠，但 baseline 在每行搬入、计算和搬出之间使用 `PIPE_ALL`，把本可并行的 MTE2/V/MTE3 强制串行化；优化的关键就是扩大 chunk、减少队列 API 与全局同步，并用事件语义保留必要的数据依赖。

**解法**：

- 去掉 CopyIn/Out 的 PIPE_ALL，交给 TQue 队列事件自动同步（快 34%）；
- 批量 chunk 路径：多块 DataCopy 一次搬入、R 留在 UB、整个 chunk 只做一次 V→S 同步；
- rstd 标量化移出向量链；weight 拷贝移出关键路径；raw TBuf 加显式 SetFlag/WaitFlag 替代队列簿记；
- tiling：blockDim=32、rowsPerChunk=8 均衡负载，逐行向量操作合并为 chunk 宽发射。

## Lab 4.5：INT8 张量核模拟 FP64 GEMM

**任务**：在 H800 MIG 上用 INT8 Tensor Core 模拟 FP64 矩阵乘（4096³ 与 8192³），在 L2 相对误差达标的前提下优化端到端吞吐。

**原理**：FP64 有 53 位有效尾数，而单个 INT8 分量只能表示有限范围，因此把每个元素 \(x\) 按残差逐级分解为 \(S\) 个 INT8 分量。令 \(r_0=x\)，则

$$
s_0=\frac{X_{\max}}{127},
\qquad
q_i=\operatorname{round}(r_i/s_i),
\qquad
r_{i+1}=r_i-q_is_i,
\qquad
s_{i+1}=\frac{s_i}{254}.
$$

于是

$$
x\approx\sum_{i=0}^{S-1}q_is_i,
\qquad q_i\in[-127,127]\cap\mathbb{Z}.
$$

比例尺缩小 254 倍，是为了覆盖上一层舍入后留下的残差，同时避免 INT8 溢出。若 \(A,B\) 分别被分解为 \(A_q^{(i)}、B_q^{(j)}\)，则矩阵乘结果为

$$
\widetilde C=
\sum_{i=0}^{S-1}\sum_{j=0}^{S-1}
s_i^As_j^B\left(A_q^{(i)}B_q^{(j)}\right).
$$

括号内使用 INT8×INT8→INT32 Tensor Core GEMM，外层缩放和累加回到 FP64。朴素实现需要 \(S^2\) 个分量对，并且还要承担逐级量化、workspace 写回和 FP64 重组开销；因此 `splits` 增大时会快速退化。实验后续把 mantissa bits 映射到 cuBLAS fixed-point emulation，由库内部融合量化、INT8 GEMM 和 FP64 重组，以减少 kernel launch 与中间结果搬运。

**解法**：

- 手写 `mma.sync` 路径：逐级量化融合为单 kernel、共享内存转置、批量 FP64 重组、CUDA Graph、按反对角线裁剪低贡献 pair；
- 量化与重组向量化（double4、int4）、重组首批直写免清零、批次扩容，非 GEMM 开销减半；
- 最终改用 cuBLAS 13 fixed-point emulation：把 splits 映射为 mantissa bits，量化、GEMM、重组全部在库内完成。

## Lab 5：Gemma4 端到端推理优化

**任务**：在 1/7 张 H800（10 GiB 显存）上基于 `hpc101_infer` 框架优化 Gemma4-12B 推理：任务一用 GPTQ 把 BF16 权重量化到 INT4（ΔNLL < 0.16），任务二最大化端到端吞吐。

**原理**：Gemma4-12B 是 decoder-only Transformer，包含 48 个 layer、隐藏维度 \(D=3840\)、词表大小 \(V=262144\)。大多数层使用窗口 \(w=1024\) 的滑动注意力，少数层使用全局注意力；KV 头数少于 Query 头数，因此可以用分组查询注意力降低 KV Cache。W4A16 下，一个形状为 \(d_{out}\times d_{in}\) 的 Linear 主要占用

$$
M_{\mathrm{linear}}=\frac{d_{out}d_{in}}{2}
 +2d_{out}\left\lceil\frac{d_{in}}{128}\right\rceil
\quad\mathrm{bytes},
$$

其中第一项是两个 INT4 code 打包进一个 byte，第二项是每 128 个输入通道保存的 FP16 scale。328 个量化 Linear 约占 5.24 GiB，tied BF16 embedding/output weight 占

$$
M_{\mathrm{embed}}=VD\times2
=2013265920\ \mathrm{bytes}\approx1.875\ \mathrm{GiB}.
$$

KV Cache 还会随 batch \(B\) 和序列长度 \(S\) 增长。若所有层都按 dense 上限分配，理论容量为

$$
M_{\mathrm{kv,dense}}=344064BS\quad\mathrm{bytes}.
$$

但 40 个滑窗层只需保存最近 \(w=1024\) 个 token，使用 ring buffer 后变为

$$
M_{\mathrm{kv,window}}=2\times2B\left[
40\times8\times256\times\min(S,1024)
 +8\times1\times512\times S\right].
$$

任务一的 GPTQ 不是独立地对每个权重做 Round-to-Nearest，而是利用校准激活 \(X\in\mathbb{R}^{N\times d_{in}}\)，线性层重构误差的 Hessian 近似为

$$
H=\frac{2}{N}X^\top X.
$$

加入阻尼后令 \(H^{-1}=U^\top U\)，其中 \(U\) 是上三角 Cholesky 因子。量化第 \(j\) 列时，先计算归一化误差，再把误差传播到后续列：

$$
e_j=\frac{w_j-q_j}{U_{j,j}},
\qquad
W_{:,j+1:}\leftarrow W_{:,j+1:}-e_jU_{j,j+1:}.
$$

这使量化误差尽量抵消校准分布上的输出误差，而不只是最小化权重 MSE。任务二的 decode 则受另一个瓶颈支配：每一步要经过 328 个 Linear，参考实现先解包 INT4、反量化并物化 BF16 权重，再调用 `F.linear`，真正的 GEMM 不足 CUDA 时间的 10%。因此优化重点依次转向 GEMV/dot 双路径、权重驻留 GPU、连续的 qweight 布局和显存安全的 KV 管理。

**解法**：

- GPTQ：两次 Cholesky 得上三角因子 U，128 列 block 内即时传播误差、block 间用矩阵乘批量传播；dead column 保住权重，ΔNLL = 0.093；
- 修 OOM：`.to()` 守卫阻止权重回流 GPU；sliding 层 KV 只开 1024 ring buffer（省约 48%）；滑窗 prefill 只写尾部窗口；
- decode Triton 双路径：M≤8 走 GEMV（合并加载加 FP32 归约），M>8 走 `tl.dot` Tensor Core；
- 固化 autotune 胜者配置，冷启动编译从 104 个 kernel 降到 7 个（749.8s → 94.5s）；
- 权重整体驻留 GPU 消除每步 H2D；qweight 转置成 K 连续布局后合并加载加 coalesced MMA，占用率 12.5%→31%。

## 结语

*Lab 4（AMSS-NCKU 数值相对论程序优化）暂未整理，待补。*
