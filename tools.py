"""Host-only source tools with bounded retry and redacted errors."""
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote
import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()
ARXIV_URL = "https://export.arxiv.org/api/query"
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"
_arxiv_lock = threading.Lock()
_last_arxiv_call = None
_ledger_lock = threading.Lock()
_source_ledger = {}

def clear_source_ledger():
    with _ledger_lock:
        _source_ledger.clear()

def source_ledger():
    """Snapshot only source metadata returned by successful host retrieval tools."""
    with _ledger_lock:
        return json.loads(json.dumps(_source_ledger))

def _remember(records, family):
    with _ledger_lock:
        for record in records:
            item = {"id": record["id"], "url": record["url"], "title": record.get("title", ""),
                    "date": record.get("published") or "undated", "source": family}
            entries = _source_ledger.setdefault(item["url"], [])
            if item not in entries:
                entries.append(item)

def _remember_web(text, requested_url=None):
    if text.startswith(("ERROR:", "NO RESULTS")):
        return
    records = []
    # A fetch proves only its requested URL. Never interpret page-body examples as metadata.
    blocks = [text.split("\n\n", 1)[0]] if requested_url else re.split(r"(?m)(?=^Title: )", text)
    for block in blocks:
        if not requested_url:
            header = re.match(r"Title: ([^\n]+)\nURL: (https?://[^\n]+)\nPublished: ([^\n]*)\nAuthor: ([^\n]*)\nHighlights:", block)
            if not header:
                continue
            block = header.group(0)
        match = re.search(r"(?m)^URL:\s*(https?://\S+)", block)
        if not match:
            continue
        url = requested_url or match.group(1)
        title = re.search(r"(?m)^Title:\s*(.+)|^#{1,6}\s+(.+)", block)
        published = re.search(r"(?m)^Published:\s*(\d{4}-\d{2}-\d{2})", block)
        records.append({"id": url, "url": url, "title": _compact(next((g for g in title.groups() if g), ""), 1000) if title else "",
                        "published": published.group(1) if published else "undated"})
    _remember(records, "web")

class RetryableError(Exception):
    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after

def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Retry RetryableError only; do not sleep after the final failure."""
    if attempts < 1 or base < 0 or cap < 0:
        raise ValueError("invalid retry settings")
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            delay = exc.retry_after
            if delay is None:
                delay = base * 2 ** attempt + random.uniform(0, base)
            time.sleep(min(cap, max(0, float(delay))))

def redact(text):
    """Remove configured secrets and URL-encoded variants."""
    text = str(text)
    for name, value in os.environ.items():
        if value and any(word in name.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            text = text.replace(value, "[REDACTED]").replace(quote(value, safe=""), "[REDACTED]")
    return re.sub(r"(?i)(exaApiKey=)[^&\s]+", r"\1[REDACTED]", text)

def _error(exc):
    return f"ERROR: {type(exc).__name__}: {redact(exc)}"

def _retry_after(value):
    if not value:
        return None
    try:
        return max(0, float(value))
    except (TypeError, ValueError):
        try:
            return max(0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            return None

def _request(method, url, **kwargs):
    try:
        response = (httpx.get if method == "GET" else httpx.post)(url, timeout=60, follow_redirects=True, **kwargs)
    except httpx.TransportError as exc:
        raise RetryableError(redact(exc)) from None
    if response.status_code in {429, 500, 502, 503, 504}:
        raise RetryableError(f"HTTP {response.status_code}", _retry_after(response.headers.get("Retry-After")))
    response.raise_for_status()
    return response

def _compact(value, length=600):
    return " ".join(str(value or "").split())[:length]

def _records(items, family=None):
    if family:
        _remember(items, family)
    return json.dumps(items, ensure_ascii=False) if items else "NO RESULTS"

@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv by short keywords, newest first. JSON: id, url, published, title, summary."""
    try:
        terms = [x for x in re.findall(r"[^\W_]+(?:-[^\W_]+)*", query) if x.upper() not in {"AND", "OR", "NOT", "ALL"}]
        if not terms:
            return "NO RESULTS"
        def fetch():
            global _last_arxiv_call
            with _arxiv_lock:
                if _last_arxiv_call is not None:
                    time.sleep(max(0, 3 - (time.monotonic() - _last_arxiv_call)))
                _last_arxiv_call = time.monotonic()
                return _request("GET", ARXIV_URL, params={"search_query": " AND ".join(f"all:{t}" for t in terms),
                    "sortBy": "submittedDate", "sortOrder": "descending", "start": 0,
                    "max_results": min(30, max(1, max_results))})
        response = with_retry(fetch, attempts=7, base=2, cap=60)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        records = []
        for entry in ET.fromstring(response.text).findall("a:entry", ns):
            identifier = re.sub(r"v\d+$", "", entry.findtext("a:id", "", ns).split("/abs/")[-1])
            if not identifier or identifier.startswith("http"):
                continue
            records.append({"id": identifier, "url": f"https://arxiv.org/abs/{identifier}",
                "published": entry.findtext("a:published", "", ns)[:10],
                "title": _compact(entry.findtext("a:title", "", ns), 1000),
                "summary": _compact(entry.findtext("a:summary", "", ns))})
        return _records(records, "arxiv")
    except Exception as exc:
        return _error(exc)

def _hf_records(items):
    records = []
    for item in items:
        paper = item.get("paper") or {}
        identifier = paper.get("id")
        if not identifier:
            continue
        records.append({"id": identifier, "url": f"https://huggingface.co/papers/{identifier}",
            "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
            "title": _compact(paper.get("title") or item.get("title"), 1000),
            "summary": _compact(paper.get("ai_summary") or item.get("ai_summary") or paper.get("summary") or item.get("summary")),
            "upvotes": paper.get("upvotes") or item.get("upvotes") or 0,
            "github": paper.get("githubRepo") or "", "stars": paper.get("githubStars") or 0})
    return records

@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Trending Hugging Face Daily Papers sorted by upvotes. Optional date YYYY-MM-DD and keyword filter. JSON: id, url, published, title, summary, upvotes, github, stars."""
    try:
        params = {"limit": min(100, max(1, limit))}
        if date:
            params["date"] = date
        records = _hf_records(with_retry(lambda: _request("GET", HF_DAILY_URL, params=params)).json())
        if keyword:
            records = [r for r in records if keyword.lower() in (r["title"] + " " + r["summary"]).lower()]
        return _records(sorted(records, key=lambda r: r["upvotes"], reverse=True), "hf-daily")
    except Exception as exc:
        return _error(exc)

@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Find Hugging Face papers by topic. JSON: id, url, published, title, summary, upvotes, github, stars."""
    try:
        if not query.strip():
            return "NO RESULTS"
        response = with_retry(lambda: _request("GET", HF_SEARCH_URL, params={"q": query, "limit": min(50, max(1, limit))}))
        return _records(_hf_records(response.json()), "hf-search")
    except Exception as exc:
        return _error(exc)

def _mcp_call(name, arguments):
    def fetch():
        key = os.getenv("EXA_API_KEY", "")
        response = _request("POST", EXA_URL, params={"exaApiKey": key} if key else {},
            headers={"Accept": "application/json, text/event-stream"},
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}})
        if response.text.lstrip().startswith("{"):
            payload = response.json()
        else:
            events = [json.loads(line[5:].strip()) for line in response.text.splitlines()
                if line.startswith("data:") and line[5:].strip() not in {"", "[DONE]"}]
            payload = next((event for event in events if "result" in event or "error" in event), None)
            if payload is None:
                raise ValueError("MCP returned no JSON-RPC result")
        if "error" in payload:
            error = payload["error"]
            if re.search(r"rate.?limit|too many requests", str(error), re.I):
                raise RetryableError(str(error))
            raise ValueError(str(error))
        result = payload.get("result", {})
        text = "\n".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
        metadata = result.get("_meta", {})
        def limited(value):
            if isinstance(value, dict):
                return any((bool(v) and re.search(r"rate.?limit", k, re.I)) or limited(v)
                           for k, v in value.items())
            if isinstance(value, list):
                return any(limited(v) for v in value)
            return isinstance(value, str) and bool(re.search(r"rate.?limit|too many requests|quota exceeded", value, re.I))
        if limited(metadata) or re.search(r"(?:exceeded|reached|hit).{0,30}rate.?limit|rate.?limit.{0,30}(?:exceeded|reached|retry)|too many requests|quota exceeded", text, re.I):
            raise RetryableError("Exa rate limit reached")
        if result.get("isError"):
            raise ValueError(text or "MCP tool failed")
        return redact(text) if text.strip() else "NO RESULTS"
    return with_retry(fetch, attempts=7, base=2, cap=60)

@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web through Exa MCP. Returns source URLs and retrieved text. State the research objective."""
    try:
        if not query.strip():
            return "NO RESULTS"
        text = _mcp_call("web_search_exa", {"query": query, "objective": objective or f"Find reliable sources about {query}",
            "numResults": min(10, max(1, num_results))})
        _remember_web(text)
        return text
    except Exception as exc:
        return _error(exc)

@tool
def web_fetch(url: str) -> str:
    """Fetch one known HTTP(S) source URL through Exa MCP, returning at most 12000 characters of page text."""
    try:
        if not re.match(r"^https?://", url):
            raise ValueError("url must use HTTP(S)")
        text = _mcp_call("web_fetch_exa", {"urls": [url], "maxCharacters": 12000})[:12000]
        _remember_web(text, url)
        return text
    except Exception as exc:
        return _error(exc)

SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]

if __name__ == "__main__":
    for fn, args in [
        (arxiv_search, {"query": "world model", "max_results": 3}),
        (hf_daily_papers, {"limit": 20}),
        (hf_search_papers, {"query": "world model", "limit": 3}),
        (web_search, {"query": "survey paper on world models", "num_results": 2}),
        (web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        print(f"== {fn.name}\n{fn.invoke(args)[:800]}\n", flush=True)
