"""Run deep research in the provided sandbox and publish validated output bytes."""
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from check_citations import check
from model import make_model
from sandbox import download, open_sandbox, upload
from tools import clear_source_ledger, redact, source_ledger
from validate_audit import validate_audit

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"
NORMALIZER_PATH = f"{WORKDIR}/normalize_sources.py"
LEDGER_PATH = f"{WORKDIR}/research/retrieved-sources.json"
AUDIT_VALIDATOR_PATH = f"{WORKDIR}/research/validate_audit.py"
FAMILIES = {"arxiv", "hf-daily", "hf-search", "web"}

def slugify(topic):
    return re.sub(r"[^\w]+", "-", topic.lower(), flags=re.UNICODE).strip("-")[:60].rstrip("-") or "topic"

def build_prompt(topic):
    daily = {
        "survey about world model": ("2024-02-26", "Genie"),
        "survey about reinforcement learning for llm reasoning": ("2025-01-23", "reasoning"),
        "survey about llm agents and tool use": ("2025-03-07", "tools"),
        "survey about video and multimodal generation": ("2024-01-19", "video"),
        "survey about efficient inference and small language models": ("2024-02-26", "MobileLLM"),
    }.get(topic.lower())
    diversity = (f"One researcher MUST call hf_daily_papers(date='{daily[0]}', keyword='{daily[1]}', limit=50), "
                 "read the actual returned evidence and include a relevant result with source=hf-daily. "
                 "The lead MUST cite that source where supported, alongside hf-search and web sources. "
                 "Do not relabel it. These three families suffice if arXiv is unavailable. " if daily else "")
    return (f"Research topic: {topic}\nToday is {time.strftime('%Y-%m-%d')}. "
            + diversity +
            "Produce an evidence-grounded English academic survey with concise, clear sentences. "
            "Use recent work from the last two years alongside foundational sources. "
            "Plan and delegate at least three independent researcher tasks in parallel. "
            "Follow the entire sandbox finalization, citation validation and spot-check workflow. "
            "Report any unresolved failure honestly; do not substitute a chat answer for output files. "
            + os.getenv("LAB_RESEARCH_FEEDBACK", ""))

def summarize(messages, elapsed, model_name):
    calls = Counter()
    delegations = Counter()
    tokens = {"input": 0, "output": 0}
    for message in messages:
        get = message.get if isinstance(message, dict) else lambda key, default=None: getattr(message, key, default)
        for call in get("tool_calls", []) or []:
            calls[call["name"]] += 1
            if call["name"] == "task":
                delegations[(call.get("args") or {}).get("subagent_type", "unknown")] += 1
        usage = get("usage_metadata", {}) or {}
        tokens["input"] += usage.get("input_tokens", 0) or 0
        tokens["output"] += usage.get("output_tokens", 0) or 0
    return {"model": model_name, "elapsed_s": round(elapsed, 1), "subagent_calls": calls["task"],
            "researcher_calls": delegations["researcher"], "citation_checker_calls": delegations["citation-checker"],
            "tool_calls": dict(calls), "tokens": tokens}

def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    audit_path = f"{WORKDIR}/research/citation-audit.json"
    files = download(backend, [REPORT_PATH, SOURCES_PATH, audit_path])
    report_bytes, sources_bytes = files.get(REPORT_PATH), files.get(SOURCES_PATH)
    if not report_bytes or not report_bytes.strip() or not sources_bytes:
        raise RuntimeError("missing or empty report/sources in sandbox")
    try:
        report = report_bytes.decode("utf-8")
        sources = json.loads(sources_bytes)
    except (UnicodeError, ValueError) as exc:
        raise RuntimeError("invalid downloaded report or sources JSON") from exc
    problems = check(report, sources)
    if problems:
        raise RuntimeError("invalid citations: " + "; ".join(problems))
    meta = {"topic": topic, **summarize(messages, elapsed, model_name), "n_sources": len(sources),
            "source_families": sorted({s.get("source", "") for s in sources})}
    if files.get(audit_path):
        meta["citation_audit"] = json.loads(files[audit_path])
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(topic)
    outputs = {f"{slug}.md": report_bytes, f"{slug}.sources.json": sources_bytes,
               f"{slug}.meta.json": (json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode()}
    # Validate all data first and stage all files before replacing output files.
    staged = []
    try:
        for name, content in outputs.items():
            with tempfile.NamedTemporaryFile(dir=reports_dir, prefix=".stage-", delete=False) as stream:
                stream.write(content)
                staged.append((Path(stream.name), reports_dir / name))
        old = {target: target.read_bytes() if target.exists() else None for _, target in staged}
        replaced = []
        try:
            for temporary, target in staged:
                os.replace(temporary, target)
                replaced.append(target)
        except OSError:
            for target in replaced:
                if old[target] is None:
                    target.unlink(missing_ok=True)
                else:
                    target.write_bytes(old[target])
            raise
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    return reports_dir / f"{slug}.md"

def _execute_ok(backend, command):
    response = backend.execute(command)
    if response.exit_code != 0:
        raise RuntimeError(f"sandbox command failed: {redact(response.output)}")
    return response.output

def _run_agent(agent, messages):
    seen = {(): len(messages)}
    final = messages
    for namespace, state in agent.stream({"messages": messages}, config={"recursion_limit": 1000},
                                        stream_mode="values", subgraphs=True):
        current = state.get("messages", [])
        for message in current[seen.get(namespace, 0):]:
            names = [call["name"] for call in getattr(message, "tool_calls", [])]
            if names:
                role = "Subagent" if namespace else "Lead"
                print(f"[research] {role} tools: " + ", ".join(names), flush=True)
        seen[namespace] = len(current)
        if not namespace:
            final = current
    return final

def _validate_sandbox(backend, messages):
    validation = _execute_ok(backend, f"python3 {VALIDATOR_PATH} --metadata")
    if not validation.startswith("OK:"):
        raise RuntimeError("sandbox validator did not print OK")
    _execute_ok(backend,
        "python3 -c 'import json; "
        f"s=json.load(open(\"{SOURCES_PATH}\")); "
        "f={x.get(\"source\") for x in s}; "
        "assert len(f & {\"arxiv\",\"hf-daily\",\"hf-search\",\"web\"}) >= 3, \"need 3 source families\"; "
        "assert all(x.get(\"source\") in {\"arxiv\",\"hf-daily\",\"hf-search\",\"web\"} and "
        "(x[\"source\"] == \"web\" or x[\"url\"] == "
        "(\"https://arxiv.org/abs/\" if x[\"source\"] == \"arxiv\" else \"https://huggingface.co/papers/\") + x[\"id\"]) for x in s), "
        "\"source family/URL/id mismatch: correct using original retrieved metadata\"'"
    )
    stats = summarize(messages, 0, "")
    if stats["researcher_calls"] < 3:
        raise RuntimeError("need at least three actual researcher delegations")
    if stats["citation_checker_calls"] < 1:
        raise RuntimeError("citation-checker was not called: delegate at least three cited claims and exact URLs")
    _execute_ok(backend, f"python3 {AUDIT_VALIDATOR_PATH}")
    return validation

def main(topic):
    topic = topic.strip()
    if not topic:
        print('Usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    started = time.monotonic()
    clear_source_ledger()
    try:
        model = make_model()
        if os.getenv("LAB_REASONING_EFFORT") and hasattr(model, "reasoning_effort"):
            model.reasoning_effort = os.environ["LAB_REASONING_EFFORT"]
        if os.getenv("LAB_USE_RESPONSES_API") == "1" and hasattr(model, "use_responses_api"):
            model.use_responses_api = True
        # Keep the provided factory unchanged; bound OpenAI transport waits and let middleware retry.
        if getattr(model, "root_client", None) is not None:
            model.root_client = model.root_client.with_options(timeout=90, max_retries=0)
            model.client = model.root_client.chat.completions
            model.root_async_client = model.root_async_client.with_options(timeout=90, max_retries=0)
            model.async_client = model.root_async_client.chat.completions
        model_name = os.getenv("LAB_MODEL") or os.getenv("OPENAI_DEPLOYMENT_MODEL") or "unknown"
        with open_sandbox() as backend:
            print(f"[research] Sandbox ready; starting {topic}", flush=True)
            _execute_ok(backend, f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
                             NORMALIZER_PATH: (ROOT / "normalize_sources.py").read_bytes(),
                             AUDIT_VALIDATOR_PATH: (ROOT / "validate_audit.py").read_bytes()})
            _execute_ok(backend, f"test -s {VALIDATOR_PATH} && test -s {FINALIZER_PATH}")
            agent = build_lead_agent(backend, model)
            messages = [{"role": "user", "content": build_prompt(topic)}]
            # Preserve lead history when requesting at most two sandbox repair passes.
            for attempt in range(3):
                messages = _run_agent(agent, messages)
                try:
                    upload(backend, {LEDGER_PATH: json.dumps(source_ledger(), ensure_ascii=False).encode()})
                    _execute_ok(backend, f"python3 {NORMALIZER_PATH}")
                    _execute_ok(backend, f"python3 {FINALIZER_PATH}")
                    validation = _validate_sandbox(backend, messages)
                    break
                except RuntimeError as exc:
                    if attempt == 2:
                        raise
                    print(f"[research] Requesting sandbox repair {attempt + 1}: {redact(exc)[:1500]}", flush=True)
                    messages = [*messages, {"role": "user", "content":
                        "Submission checks failed: " + redact(exc) +
                        "\nFix these issues in the sandbox using retrieved evidence. Do not invent metadata. "
                        "Rerun finalizer and validator after edits; recheck three source families. "
                        "If arXiv remains unavailable, use relevant historical hf_daily_papers dates. "
                        "For example, reasoning papers are available on 2025-01-23; choose dates relevant to your topic. "
                        "Use citation-checker for at least three claims and save its actual verdicts to "
                        f"{WORKDIR}/research/citation-audit.json as an array of claim,url,verdict,evidence,correction objects. "
                        "Use exact sentences from the current report and final source URLs. All three must be SUPPORTED. "
                        "Do not claim completion until resolved."}]
            path = save_outputs(backend, topic, messages, time.monotonic() - started, model_name)
            print(validation.strip(), flush=True)
        print(f"Saved: {path}", flush=True)
        return 0
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {redact(exc)}", file=sys.stderr, flush=True)
        cause = exc.__cause__
        if cause is not None:
            print(f"CAUSE: {type(cause).__name__}: {redact(cause)}", file=sys.stderr, flush=True)
        return 1

if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
