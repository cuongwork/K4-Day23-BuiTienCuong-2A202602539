# World Models for Sequential Decision-Making and Interactive Simulation

## TL;DR
- Early world-model work defined the field as learned internal simulators that compress observation histories, predict latent futures, and support planning or policy learning inside imagination [1][2][3].
- Genie extends this idea toward interactive video worlds: the paper describes unsupervised training from unlabelled video, spatiotemporal tokenization, autoregressive dynamics, and frame-by-frame interaction via a latent action model [4].
- Recent evaluation work shifts attention away from next-frame loss alone toward planning success, counterfactual discrimination, instruction following, physics adherence, and symbolic generation tasks [5][6][7].
- The 2025-2026 literature increasingly treats action-conditioned video generation as a practical world-model interface, but the retrieved summaries do not yet provide enough detail to verify broad benchmark dominance or a single standard metric [8][9][10].

## Background
World models began as a compact alternative to direct policy learning from pixels. The foundational papers define a learned simulator that encodes observations into a latent state, predicts future latent states under action, and lets an agent plan either in the learned model or from imagined rollouts [1][2][3]. The key contribution was not merely prediction. It was the coupling of prediction with control: the model should produce a representation that is useful for decision-making, not only visually plausible [1][3].

The core design pattern has remained stable. First, compress sensory input into a latent state. Second, learn action-conditioned dynamics. Third, attach task heads such as reward or termination predictors when the model is used for reinforcement learning. Fourth, use the learned dynamics for planning, policy optimization, or counterfactual simulation [1][2][8]. Later survey writing keeps this framing and describes world models as internal simulators that allow agents to predict, plan, and reason within learned representations [8].

## From latent dynamics to interactive environments
Genie marks a shift from world models for control to world models as interactive environments. The retrieved HF daily paper card explicitly identifies Genie as an unsupervised generative model that creates action-controllable virtual worlds from unlabeled videos using spatiotemporal tokenization and autoregressive dynamics [4]. The web abstract confirms the same high-level architecture and adds that Genie is trained from Internet video and enables frame-by-frame interaction without ground-truth action labels [11]. That combination matters because it decouples interaction from manual action annotation and instead derives a latent action space from video structure [4][11].

This is a different training regime from classical model-based RL. The classical line assumes the agent observes actions taken in the environment, then learns a dynamics model from those interactions [1][3]. Genie instead treats video as the main supervision signal and infers latent actions post hoc [4][11]. The practical implication is that a world model can be built from passive media, not only from embodied rollouts. The retrieved evidence presents Genie as the first generative interactive environment and a foundation world model [11].

## What recent work changes in training objectives
Recent papers move the center of gravity from pure prediction to controllability and rollout stability. The 2026 summaries suggest that current systems often combine a latent dynamics backbone with a decoder that restores visual detail, or add action-conditioned modules to improve causal generation and dynamic consistency [9][10]. This indicates a recurring architectural split: latent state for reasoning and planning, plus a renderer or diffusion decoder for producing high-quality videos [9][10]. In other words, recent systems increasingly separate “world state” from “appearance.”

The same pattern appears in survey language. One retrieved survey excerpt states that recent world-model research emphasizes action-conditioned future prediction and applications such as robotics and autonomous driving [8]. Another survey excerpt characterizes interactive world models as systems that incorporate user actions into world-state transitions [11]. The available evidence therefore supports a broad convergence around action-conditioning, long-horizon rollout, and multimodal interfaces, especially in video-centric settings [11][8][9][10].

## Evaluation: from prediction quality to task success
Evaluation has become more heterogeneous. Traditional metrics focus on prediction error or perceptual fidelity, but the retrieved evidence shows a broader view. WorldPrediction uses discriminative multiple-choice tests to distinguish the correct action or action sequence from counterfactual distractors, and separates single-step world-model judgment from long-horizon procedural planning [7]. WorldModelBench evaluates video generation models as world models using instruction-following and physics-adherence metrics [5]. Text2World extends benchmarking into symbolic generation with a PDDL-based setup and multi-criteria scoring [6].

These benchmarks share an important intuition: a world model should be judged by whether it supports the right downstream behavior, not only by whether it predicts pixels well [5][6][7][8]. That is also consistent with the foundational literature, where the model is valuable because it improves control, planning, or policy learning [1][2][3]. Yet the benchmarks also expose unresolved tension. A system may achieve visually plausible prediction without reliable long-horizon planning; conversely, task success can hide weak internal dynamics [7][8]. This tension is now central to the field.

## Trends and open problems
The clearest trend is the move toward interactive, action-conditioned world models that can be trained from video, language, or mixed multimodal data [4][11][9][10]. Another trend is evaluation diversification: the field now uses planning tasks, counterfactual discrimination, symbolic world generation, and judge-based scoring for physics or instruction adherence [5][6][7]. A third trend is architectural decomposition. Recent systems increasingly separate latent reasoning from visual realization, often through a dynamics backbone plus a video decoder or diffusion module [9][10].

Several open problems remain visible in the retrieved evidence. First, long-horizon consistency is still unresolved. Surveys and benchmarks repeatedly stress horizon limits, memory, and rollout stability [7][8][9]. Second, partial observability and shortcut cues remain obstacles, especially when benchmarks try to test true causal understanding rather than superficial continuity [7]. Third, the field lacks a single evaluation standard that works across visual, symbolic, and embodied settings [5][6][8]. Finally, the video-only promise of systems like Genie is compelling, but the retrieved summaries do not yet establish how far latent-action control transfers across domains or whether such models scale cleanly beyond the reported settings [4][11].

## References
[1] World Models. arxiv. https://arxiv.org/abs/1803.10122 (2018-03-27)
[2] Recurrent World Models Facilitate Policy Evolution. arxiv. https://arxiv.org/abs/1809.01999 (2018-09-04)
[3] Learning Latent Dynamics for Planning from Pixels. arxiv. https://arxiv.org/abs/1811.04551 (2018-11-12)
[4] Genie: Generative Interactive Environments. hf-daily. https://huggingface.co/papers/2402.15391 (2024-02-23)
[5] WorldModelBench: Judging Video Generation Models As World Models. hf-search. https://huggingface.co/papers/2502.20694 (2025-02-28)
[6] Text2World: Benchmarking Large Language Models for Symbolic World Model Generation. hf-search. https://huggingface.co/papers/2502.13092 (2025-02-18)
[7] WorldPrediction: A Benchmark for High-level World Modeling .... web. https://arxiv.org/abs/2506.04363 (undated)
[8] Understanding World or Predicting Future? A Comprehensive Survey of World Models | ACM Computing Surveys. web. https://dl.acm.org/doi/10.1145/3746449 (2025-09-09)
[9] Persistent Robot World Models: Stabilizing Multi-Step Rollouts via Reinforcement Learning. hf-search. https://huggingface.co/papers/2603.25685 (2026-03-26)
[10] VideoWorld 2: Learning Transferable Knowledge from Real-world Videos. hf-search. https://huggingface.co/papers/2602.10102 (2026-02-10)
[11] Genie: Generative Interactive Environments. web. https://proceedings.mlr.press/v235/bruce24a.html (2024-07-08)
