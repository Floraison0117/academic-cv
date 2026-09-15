---
title: "Parallel Compute: A Case study of Speculative Decoding"
summary: "This note surveys five parallelism strategies for large-scale computing. It then follows speculative decoding from vanilla draft-then-verify and EAGLE to diffusion-based drafting, DSpark, DFlash, and K-forcing."
date: 2026-09-15
lastmod: 2026-09-15
authors:
  - me
tags:
  - 并行计算
  - LLM serving
  - Speculative Decoding
  - Diffusion Language Models
toc: true
---

<div style="width: 100%; margin: 2rem 0;">
  <iframe src="/pdfs/parallel-compute-speculative-decoding.pdf" title="Parallel Compute: A Case study of Speculative Decoding" style="display: block; width: 100%; height: 80vh; min-height: 720px; border: 1px solid rgba(100, 116, 139, 0.25); border-radius: 0.75rem; background: #f8fafc;" loading="lazy"></iframe>
</div>

Five common parallelism strategies are widely used in large-scale parallel computing: *data parallelism, tensor parallelism, pipeline parallelism, expert parallelism (MoE), and context parallelism*. Data parallelism replicates the model across devices and partitions the input batch; tensor parallelism shards individual tensor operations across multiple devices; pipeline parallelism divides model layers into stages that process different micro-batches concurrently; expert parallelism distributes different experts in a MoE model across devices and dynamically routes tokens among them; and context parallelism partitions long input sequences across devices to distribute the computation and memory cost of attention.

Conceptually, these strategies are relatively straightforward. In practice, however, they are system-level optimizations deeply intertwined with *asynchronous communication, data movement, synchronization, memory management, and workload scheduling*. Their actual performance depends not only on how computation is partitioned, but also on how effectively communication can be overlapped with computation and how well synchronization and load imbalance are controlled. Consequently, implementing and optimizing these seemingly simple parallelism strategies can become substantially more complicated at scale.

These are system-level optimization. As for algorithm-level design, Speculative Decoding one of the most prevalent design in LLM serving system. 

The motivation of SD is: a target model can score all positions of the drafted sequence in parallel, although the draft itself may be sequential. Therefore, we introduce the draft-then-verify paradigm. The drafter is a light weight model that decodes swiftly and the LLM just verify and correct for one time. 

For vanilla SD, the limitations are as follow:
- It requires an additional draft model and draft-target mismatch limits acceptance rate;
- Extra memory footprint is demanded, and decoding is always a memory-bandwidth bound. 

EAGLE families are famous and outstanding for multiple-token prediction. EAGLE-1 predict target-model features insted of direct;y drafting tokens, thus a single layer is qualified for prediction. In EAGLE-2, the fixed-shape tree is replaced by dynamically allocated draft tree according to predicted acceptance probabilities. EAGLE-3 further leverage multi-level target-model features for more accurate feature prediction. 

So how do we ensure lossless verification and resampling? Here, we adopt rejection sampling to preserve the target model distribution. Tree attention artfully letevery token only attends its ancestors, and thus a single turn of rejection sampling is enough. 

As for training, EAGLE-1/2 introduce two losses: feature level and token level. EAGLE-3 removes the feature-prediction constraint and improves drafting with direct token prediction, multi-level target-feature fusion, and training-time test. 

However, autoregression is inherently sequential. How about introducing diffusion models?

DSpark removes the autoregressive bottleneck of speculative drafting by generating a block of draft tokens in parallel with a diffusion model. 

Dflash accelerates inference by injecting cached target-model features into every draft layer and replacing autoregressive token-by-token drafting with a single parallel masked-block forward pass. 

DFlash trains a lightweight diffusion drafter on randomly anchored masked blocks, conditioned on frozen target-model features via KV injection, using position-decayed cross-entropy to prioritize early speculative tokens. 

DSpark aims to retain DFlash's parallel drafting efficiency while recovering token-to-token consistency and avoiding costly verification of low-confidence suffix tokens. It introduces lightweight causal dependency, usually Markov or RNN.

On system level, DSpark predicts survival probability of each draft prefix and uses a hardware-aware scheduler to verify only prefixes whose expected acceptance gain justifies their target-model batch cost, pruning low confidence suffixes before expensive verification.  

draft-then-verify methods are effective, but they have some disadvantages:
- Verification becomes expensive;
- Rejected tokens waste substantial compute;
- Variable acceptance creates ragged batching.

Diffusion Language models can predict multiple future positions in parallel, but they model per-position marginals rather than the joint future-token distribution, so parallel sampling generally breaks token dependencies and cannot reduce NFEs losslessly.

K-forcing aims to predict the joint distribution of next k tokens. In order to predict the joint distribution, they put forward the push-forward language model. Noise Inversion, as the baseline, is affected by train-inference mismatch and numerical fragility. The solution in k-forcing is progressive self-forcing distillation. 

With regard to architecture design, AR teacher backbone compresses all noise variables into one conditioning token and decode k future positions with independent heads. A newly added architecture represent each noise variables as a separate causal token, so the latter token depends on the former tokens.

In conclusion, we talk about five parallelism strategies, the vanilla speculative decoding, EAGLE families as representitive of draft-then-vefify paradigm, Dflash, DSpark introduce diffusion-LLM in multiple-token prediction and k-forcing is more batch-serving friendly.
