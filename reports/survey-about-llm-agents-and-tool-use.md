# LLM Agents and Tool Use: A Survey

## TL;DR
- Tool use in LLM agents has shifted from static API selection toward interleaved reasoning, stateful interaction, and benchmarked tool invocation. ReAct established the reason-act template, while later systems such as Toolformer and START focus on how models learn to call tools and when they do so [1][2][3].
- Recent benchmarks stress that evaluation design matters. UltraTool, GTA, ToolSandbox, AgentRewardBench, and MCPToolBench++ each show that simplified or off-policy tests can miss planning, state, multimodality, or automatic evaluation failure modes [4][5][6][7][8].
- The strongest recent trend is to couple training objectives with tool behavior. START uses hint-based tool activation and rejection-based self-training, while Toolformer and related work treat tool calls as learnable text events rather than external wrappers [3][2].
- Survey sources converge on a common taxonomy: interleaved reasoning and action, self-supervised data generation, and evaluation. This taxonomy is useful because the field now spans both how agents are trained and how they are scored [9][10].

## Background
LLM agents with tool use are best understood as systems that combine language generation with external operations such as search, calculation, APIs, and environment actions. The foundational ReAct formulation made this coupling explicit by alternating reasoning traces and task-specific actions. Its value for the field is conceptual as well as practical: it showed that action can support information gathering while reasoning can maintain and revise a plan [1]. Toolformer then pushed the idea toward self-supervised tool learning, arguing that language models can teach themselves when to call APIs and what arguments to pass [2]. These papers established the basic design space that later work continues to extend.

Recent surveys organize the field into recurring paradigms rather than isolated models. One survey frames tool use, planning, and feedback learning as the core agentic families, while another survey splits the literature into interleaved reasoning and action, self-supervised data generation, and evaluation [9][10]. This framing is useful because the current literature is no longer about tool calls alone. It also includes how models are trained to observe, decide, act, and recover from errors across multi-step trajectories [9][10].

## Learning to call tools
A central question is whether tool use should be learned from demonstrations, self-generated traces, or lightweight prompting. Toolformer argues for self-supervision: tool calls are inserted into text, and the model learns when they improve downstream likelihood [2]. START extends this direction for reasoning-heavy settings. The paper page for START reports that START integrates external tools into large reasoning models using Hint-infer and Hint Rejection Sampling Fine-Tuning [3]. The arXiv abstract similarly describes START as a tool-integrated long CoT model that uses hints to stimulate tool use without demonstration data and then filters and modifies reasoning trajectories before fine-tuning [3].

This line of work is important because it treats tool invocation as part of the model's own learning dynamics. It also suggests that the training objective is not merely to answer correctly, but to produce trajectories that can search, compute, self-check, and debug. START's reported benchmarks include math, science, and code tasks, which makes it relevant to agentic reasoning rather than only retrieval or command completion [3].

## Evaluation and benchmark design
The evaluation literature increasingly argues that tool-use competence is domain- and protocol-sensitive. UltraTool was built because existing benchmarks often use simple synthesized queries and do not reflect real-world complexity; it instead covers planning, creation, and usage across many tools and domains [4]. GTA makes a similar argument from another angle, noting that AI-generated queries, single-step tasks, dummy tools, and text-only interactions obscure real agent behavior. It therefore uses real user queries, real deployed tools, and multimodal inputs [5].

More recent benchmarks focus on properties that ordinary task success measures miss. ToolSandbox emphasizes stateful tool execution, implicit dependencies between tools, and on-policy conversational evaluation [6]. AgentRewardBench shifts attention to automatic evaluation of web-agent trajectories and warns that rule-based methods may underreport success [7]. MCPToolBench++ targets the emerging Model Context Protocol ecosystem and frames evaluation around diverse datasets and tool calls [8]. Together these benchmarks show that tool use is not a single skill. It includes planning, state tracking, multimodal perception, and reliable scoring [4][5][6][7][8].

## Recent directions in agent behavior
A notable recent theme is deciding when to call a tool, not only how to call it. The HF-search result on LLM Agents Already Know When to Call Tools -- Even Without Reasoning reports a benchmark that identifies conditions under which tool calls are necessary and finds a gap between latent prediction and actual execution [11]. That result complements the broader benchmark trend: models may recognize the need for assistance but still fail to enact it at the right moment [11].

Another theme is asynchronous and multi-task tool use. AsyncTool frames asynchronous function calling as a coordination problem under delayed responses, which matters for agents that must manage several tool requests simultaneously [12]. This direction extends the field beyond single-turn function calling toward systems that behave more like schedulers or operators. It also connects to the evaluation emphasis on state, latency, and interaction history [12][6].

## Trends and open problems
The field now appears to be converging on three intertwined problems: learning tool policies, evaluating them faithfully, and making them robust in stateful environments. On the learning side, self-training and self-generated traces remain attractive because they reduce annotation burden, but they can amplify errors if the model's own traces are poor [3][2]. On the evaluation side, benchmarks increasingly reject simplified settings and instead model real queries, hidden state, conversation history, and heterogeneous tools [5][4][6]. On the systems side, asynchronous and multi-task settings show that tool use is becoming a coordination problem, not only a prompting problem [12].

Two open problems recur across the sources. First, evaluation metrics lag behind practical behavior. Several benchmarks warn that existing protocols undercount success or fail to expose stateful failures [7][5][6]. Second, training signals remain weakly coupled to deployment settings. START and Toolformer suggest that models can learn to use tools from text-based traces, but it is still unclear how well those traces transfer to real APIs, real latency, and real user interaction [3][2]. The literature therefore points toward richer interaction data, stronger stateful benchmarks, and clearer separation between latent capability and actual tool execution [9][10][12][11].

## References
[1] ReAct: Synergizing Reasoning and Acting in Language Models. web. https://arxiv.org/abs/2210.03629 (2023-03-10)
[2] Toolformer: Language Models Can Teach Themselves to Use Tools. web. https://proceedings.neurips.cc/paper_files/paper/2023/file/d842425e4bf79ba039352da0f658a906-Paper-Conference.pdf (undated)
[3] START: Self-taught Reasoner with Tools. hf-daily. https://huggingface.co/papers/2503.04625 (2025-03-06)
[4] Planning, Creation, Usage: Benchmarking LLMs for Comprehensive Tool Utilization in Real-World Complex Scenarios. web. https://arxiv.org/abs/2401.17167 (2024-06-03)
[5] GTA: A Benchmark for General Tool Agents. web. https://proceedings.neurips.cc/paper_files/paper/2024/file/8a75ee6d4b2eb0b777f549a32a5a5c28-Paper-Datasets_and_Benchmarks_Track.pdf (undated)
[6] ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities. web. https://arxiv.org/abs/2408.04682 (undated)
[7] AgentRewardBench: Evaluating Automatic Evaluations of Web Agent Trajectories. hf-search. https://huggingface.co/papers/2504.08942 (2025-04-11)
[8] MCPToolBench++: A Large Scale AI Agent Model Context Protocol MCP Tool Use Benchmark. hf-search. https://huggingface.co/papers/2508.07575 (2025-08-11)
[9] A Review of Prominent Paradigms for LLM-Based Agents: Tool Use (Including RAG), Planning, and Feedback Learning. web. https://arxiv.org/html/2406.05804 (undated)
[10] Agentic Tool Use in Large Language Models: A Survey. web. https://arxiv.org/html/2604.00835 (2026-06-29)
[11] LLM Agents Already Know When to Call Tools -- Even Without Reasoning. hf-search. https://huggingface.co/papers/2605.09252 (2026-05-10)
[12] AsyncTool: Evaluating the Asynchronous Function Calling Capability under Multi-Task Scenarios. hf-search. https://huggingface.co/papers/2605.27995 (2026-05-27)
