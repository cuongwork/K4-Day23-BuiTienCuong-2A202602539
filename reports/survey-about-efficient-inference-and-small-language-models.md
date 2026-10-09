# Efficient Inference and Small Language Models: A Survey

## TL;DR
- Efficient inference for language models now combines model-side compression with system-side serving improvements. Surveys consistently group the field around quantization, distillation, compact architectures, speculative decoding, and KV-cache management [1][2][3].
- Small language model (SLM) work has become more evaluation-aware. Recent surveys report not only task accuracy but also prefill latency, decode speed, memory footprint, and energy use on edge devices [4][5][6].
- MobileLLM exemplifies the sub-billion-parameter, on-device direction. The paper abstract describes deep and thin architectures with embedding sharing and grouped-query attention for mobile deployment [7][8].
- Serving systems matter even when models are small. PagedAttention and related KV-cache methods target fragmentation and memory pressure, which shape throughput and batching behavior during inference [9][1].

## Background

Efficient inference is no longer treated as a single optimization problem. Recent surveys organize the area into a stack of approaches: parameter reduction, activation and weight compression, decoding-time acceleration, and memory-aware serving [1][2]. This taxonomy matters because the main bottlenecks differ by setting. Mobile deployment is constrained by RAM, energy, and latency. Datacenter deployment is often constrained by throughput, batching efficiency, and KV-cache growth [9][1].

The SLM literature reflects the same shift. Surveys of small models describe them as practical for on-device and edge use, but they also emphasize that evaluation must include runtime costs alongside accuracy [4][5]. This is an important change from older compression work, which often emphasized model size or task score alone. Distillation is a foundational example: DistilBERT is described as smaller, faster, and cheaper while preserving most language understanding capability, and it established compression as a serious research program rather than a mere engineering trick [10].

## Model compression and architectural efficiency

A central line of work reduces inference cost by changing the model itself. Survey evidence shows that quantization remains one of the dominant tools because it cuts memory use and compute by lowering precision for parameters, activations, or gradients [1]. This is particularly relevant for SLMs, where the target is often not only fewer FLOPs but also compatibility with constrained devices [4]. The low-bit literature frames this as a systems problem as much as an algorithmic one, because the effective cost of a model depends on how weights and activations map onto hardware memory hierarchies [1].

Architectural efficiency complements quantization. The MobileLLM paper argues for deep, thin architectures, embedding sharing, and grouped-query attention as a recipe for sub-billion-parameter models aimed at mobile use [7][8]. The abstract further reports gains over prior 125M and 350M models while keeping size fixed for one variant and only marginally increasing latency [8]. This matters because the best efficiency point is not always the smallest parameter count; rather, it is a balance among depth, width, attention design, and memory reuse [7][8]. Recent survey work on SLMs places these model-design choices alongside pruning and quantization in a broader compression taxonomy [4].

Distillation remains relevant as a bridge between large teacher models and compact students. The DistilBERT summary shows why: it preserves much of the original model’s capability while reducing compute and latency [10]. Although DistilBERT is not a decoder-only LLM, it provides an enduring template for compression by supervision transfer. Recent SLM surveys continue to treat distillation as a core training method, alongside pruning and quantization, because small models often need task-specific adaptation after compression [4].

## Decoding-time acceleration

A second research family accelerates inference without changing the target model’s weights. Speculative decoding is the clearest example. Survey summaries describe it as drafting multiple tokens with a smaller model and then verifying them with the larger model [2][3]. This changes generation from strictly sequential decoding to a draft-and-check process [2][3]. The method is attractive because it can exploit small draft models that are already useful in SLM settings, but its efficiency depends on how often the draft is accepted and how expensive verification is [2][3].

The speculative-decoding literature is therefore best read as a complement to model compression rather than a substitute. A compact draft model is itself easier to deploy, but the overall speedup comes from coordinating the draft model, the verifier, and the token acceptance rule [2][3]. In practice, this makes throughput a joint property of model quality and decoding strategy. Survey work in this area also suggests that the relevant design space is still moving, with recent reviews focusing on the draft-verification interface rather than a single canonical algorithm [3].

## Memory, caching, and serving systems

Model-side efficiency does not remove the importance of serving infrastructure. KV-cache management has become a central optimization because cached keys and values grow with sequence length and can dominate memory use [1][9]. The PagedAttention paper proposes an attention algorithm inspired by virtual memory and paging that reduces KV-cache waste and fragmentation [9]. The broader serving literature argues that this enables near-zero waste in cache memory and more flexible sharing across requests [9].

This system-level view is essential for SLMs as well. The usual intuition is that small models are automatically easy to serve, but recent evidence is more nuanced. Survey work on SLMs reports measurements of prefill and decode speed, memory footprint, and energy consumption on devices such as Jetson Orin and smartphones [4][5]. Another study of edge deployment emphasizes that dynamic task-specific routing, model-hardware co-design, and vocabulary/KV-cache compression remain open opportunities [6]. These claims suggest that SLM deployment is limited not only by parameter count but also by cache behavior, batching policy, and hardware fit [5][6][9].

## Evaluation of small language models

Evaluation practices for SLMs are broadening. Recent surveys, including this one, evaluate capabilities such as commonsense reasoning, mathematics, in-context learning, and long context, and benchmark on-device runtime costs such as inference latency and memory footprints [5]. This is a useful correction to earlier compression work, because a small model that is fast but weak on reasoning may not be acceptable for practical deployment. The evaluation space is therefore multi-objective: capability, latency, memory, and energy all matter [4][5].

The current literature also shows that size alone is an incomplete proxy for deployability. The survey of SLMs reports a 100M–5B parameter range and highlights device classes from smartwatches to tablets [5]. The edge-deployment paper goes further by suggesting that some SLMs can outperform 7B models on general tasks while still showing limited in-context learning [6]. This implies that small models can be competitive on some task families without solving all capabilities that matter for interactive use [6]. MobileLLM fits into this same picture by focusing on a sub-billion-parameter regime and on-device use cases rather than generic benchmark maximization [7][8].

## Trends and open problems

The field is converging on a layered view of efficiency. First, model design controls the baseline cost through width, depth, attention structure, and compression [7][8][1]. Second, decoding-time methods such as speculative decoding reduce generation cost when small draft models are available [2][3]. Third, serving systems determine whether those algorithmic gains translate into real throughput under memory constraints [9][1]. The most useful recent surveys treat these as interacting components rather than isolated techniques [1][4][5].

Open problems are increasingly methodological. Benchmarks still differ in whether they emphasize raw task quality or deployment metrics such as decode speed, energy, and memory [4][5]. The evidence on edge deployment also suggests that in-context learning and long-context behavior remain weak points for many SLMs [6]. For MobileLLM-like systems, the main question is not whether a sub-billion model can be built, but which combination of architecture, quantization, and cache management yields the best tradeoff for a specific device class [7][8][9]. For speculative decoding, the open issue is how to make draft-verification pipelines robust across heterogeneous model pairs and workloads [2][3]. Across the literature, efficient inference is therefore moving from single-technique optimization toward co-designed model, decoding, and serving stacks [1][5][9].

## References
[1] Model Compression and Efficient Inference for Large Language Models: A Survey. arxiv. https://arxiv.org/abs/2402.09748 (2024-02-15)
[2] Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding. hf-search. https://huggingface.co/papers/2401.07851 (2024-01-15)
[3] Closer Look at Efficient Inference Methods: A Survey of Speculative Decoding. hf-search. https://huggingface.co/papers/2411.13157 (2024-11-20)
[4] A Survey of Small Language Models. hf-search. https://huggingface.co/papers/2410.20011 (2024-10-25)
[5] Small Language Models:Survey, Measurements, and Insights. web. https://arxiv.org/abs/2409.15790 (2025-02-26)
[6] Demystifying Small Language Models for Edge Deployment. web. https://aclanthology.org/anthology-files/anthology-files/pdf/acl/2025.acl-long.718.pdf (undated)
[7] MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases. hf-daily. https://huggingface.co/papers/2402.14905 (2024-02-22)
[8] MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases. web. https://proceedings.mlr.press/v235/liu24ce.html (2024-07-08)
[9] Efficient Memory Management for Large Language Model .... web. https://dl.acm.org/doi/10.1145/3600006.3613165 (2023-10-23)
[10] DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. hf-search. https://huggingface.co/papers/1910.01108 (2019-10-02)
