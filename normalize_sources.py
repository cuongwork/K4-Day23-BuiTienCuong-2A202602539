"""Normalize metadata against actual host tool results, exclusively inside the sandbox."""
import json
import re
import sys
from collections import Counter
from urllib.parse import urlparse

SOURCES = "/tmp/work/research/sources.json"
LEDGER = "/tmp/work/research/retrieved-sources.json"

def _paper_id(url):
    parsed = urlparse(url)
    if parsed.netloc not in {"arxiv.org", "huggingface.co"}:
        return None
    match = re.fullmatch(r"/(?:abs|html|pdf|papers)/(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?/?", parsed.path)
    return match.group(1) if match else None

def normalize(sources, ledger):
    """Return normalized copies and problems; never invent URLs, IDs, titles, dates or labels."""
    if not isinstance(sources, list) or not isinstance(ledger, dict):
        return sources, ["expected sources array and retrieval ledger object"]
    normalized, problems, counts = [], [], Counter()
    for entry in sources:
        if not isinstance(entry, dict):
            problems.append("source entry is not an object")
            continue
        candidates = ledger.get(entry.get("url"), [])
        if not candidates and isinstance(entry.get("url"), str):
            identifier = _paper_id(entry["url"])
            if identifier:
                candidates = [record for url, records in ledger.items() if _paper_id(url) == identifier
                              for record in records]
        if not candidates:
            problems.append(f"source [{entry.get('n')}] URL not found in successful retrieval results: {entry.get('url')}. "
                            "Ask researcher to retrieve this exact source, or replace it with an actually retrieved source.")
            continue
        preferred = [record for record in candidates if record["source"] == entry.get("source")]
        chosen = min(preferred or candidates, key=lambda record: (counts[record["source"]], record["source"], record["url"]))
        counts[chosen["source"]] += 1
        normalized.append({"n": entry.get("n"), **chosen})
    return normalized, problems

def main():
    try:
        with open(SOURCES, encoding="utf-8") as stream:
            sources = json.load(stream)
        with open(LEDGER, encoding="utf-8") as stream:
            ledger = json.load(stream)
        result, problems = normalize(sources, ledger)
        if problems:
            print("\n".join(problems))
            return 1
        with open(SOURCES, "w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
        print(f"NORMALIZED: {len(result)} sources grounded in host tool results")
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"cannot normalize sources: {exc}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
