"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    PSEUDO-CODE:
      problems = []
      if sources is empty: return ["no sources in sources.json"]
      for each source entry:
          n must be an int                       -> problem if not
          url must start with http:// or https://-> problem if not
          the same url must not appear twice     -> problem if duplicated
      split report_text at the heading "## References":
          body = text before it; if the heading is missing -> problem
      cited = set of numbers found as [n] in the BODY only (not in the reference list; use a regex)
      every number in `cited` must exist in sources -> problem "[n] cited but missing from sources.json"
      every source number must be in `cited`        -> problem "source [n] never cited"
      the lines of the References section that start with "[n]" (regex) are the reference lines:
          every source needs exactly ONE reference line (none missing, no number twice, no number that is not a source)
          each reference line holds exactly ONE http(s) URL and it must equal that source's url
          (a line bundling several sources under one number is a problem)
      return problems
    """
    problems = []
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json (expected a nonempty array)"]
    by_number, urls = {}, set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            problems.append(f"source entry {index + 1} is not an object")
            continue
        n, url = source.get("n"), source.get("url")
        if type(n) is not int or n < 1:
            problems.append(f"source entry {index + 1}: n must be a positive integer")
        elif n in by_number:
            problems.append(f"duplicate source number [{n}]")
        else:
            by_number[n] = source
        if not isinstance(url, str) or not re.fullmatch(r"https?://[^\s]+", url):
            problems.append(f"source entry {index + 1}: invalid URL")
        elif url in urls:
            problems.append(f"duplicate source URL: {url}")
        else:
            urls.add(url)
    # Ignore fenced/inline code; neither code examples nor Markdown links are citations.
    clean = re.sub(r"(?ms)^\s*(`{3,}|~{3,})[^\n]*\n.*?^\s*\1\s*$", "", report_text)
    clean = re.sub(r"`+[^`\n]*`+", "", clean)
    clean = re.sub(r"(?m)^(?: {4}|\t)[^\n]*(?:\n|$)", "", clean)
    headings = list(re.finditer(r"(?m)^## References\s*$", clean))
    if len(headings) != 1:
        problems.append("report must contain exactly one ## References heading")
    body = clean[:headings[0].start()] if headings else clean
    references = clean[headings[0].end():] if headings else ""
    if re.search(r"(?m)^#{1,6}\s+", references):
        problems.append("References must be the final section")
    cited = set()
    for match in re.finditer(r"(?<!!)\[(\d+(?:\s*[-,]\s*\d+)*)\](?!\s*\()", body):
        for part in match.group(1).split(","):
            bounds = [int(x.strip()) for x in part.split("-")]
            if len(bounds) == 1:
                cited.add(bounds[0])
            elif bounds[0] <= bounds[1] and bounds[1] - bounds[0] <= 10000:
                cited.update(range(bounds[0], bounds[1] + 1))
            else:
                problems.append(f"invalid citation range: {match.group(0)}")
    for n in sorted(cited - by_number.keys()):
        problems.append(f"[{n}] cited but missing from sources.json")
    for n in sorted(by_number.keys() - cited):
        problems.append(f"source [{n}] never cited")
    reference_counts = {}
    for line in references.splitlines():
        if not line.strip():
            continue
        match = re.match(r"^\[(\d+)\]\s+(.+)$", line.strip())
        if not match:
            problems.append("invalid reference line: " + line[:120])
            continue
        n = int(match.group(1))
        reference_counts[n] = reference_counts.get(n, 0) + 1
        if n not in by_number:
            problems.append(f"reference [{n}] missing from sources.json")
        found_urls = re.findall(r"https?://[^\s<>]+", match.group(2))
        if len(found_urls) != 1:
            problems.append(f"reference [{n}] must contain exactly one URL")
        elif n in by_number and found_urls[0] != by_number[n].get("url"):
            problems.append(f"reference [{n}] URL differs from sources.json")
    for n in sorted(by_number):
        if reference_counts.get(n, 0) != 1:
            problems.append(f"source [{n}] needs exactly one reference line")
    return problems


def check_structure(report_text):
    """Check the required survey layout without evaluating scientific truth."""
    required = {"TL;DR", "Background", "Trends and open problems", "References"}
    headings = re.findall(r"(?m)^##[ \t]+(.+?)[ \t]*$", report_text)
    problems = [f"missing required section: {name}" for name in sorted(required - set(headings))]
    themes = [name for name in headings if name not in required]
    if not 3 <= len(themes) <= 6:
        problems.append("report needs 3 to 6 thematic sections")
    background = re.search(r"(?ms)^## Background[ \t]*\n(.*?)(?=^## |\Z)", report_text)
    if background and not re.search(r"\[\d+\]", background.group(1)):
        problems.append("Background needs a citation to retrieved foundational evidence")
    if not re.match(r"^#\s+\S", report_text):
        problems.append("report needs a title")
    body = report_text.split("## References", 1)[0]
    for paragraph in re.split(r"\n\s*\n|(?m:^#{1,6}[^\n]*\n)", body):
        prose = "\n".join(line for line in paragraph.splitlines() if not line.startswith("#"))
        if len(prose.split()) >= 8 and not re.search(r"\[\d+(?:\s*[-,]\s*\d+)*\]", prose):
            problems.append("substantive paragraph needs supporting citations: " + prose[:120])
    return problems


def check_source_metadata(sources):
    """Explain source-family contract errors precisely so the lead can repair them."""
    if not isinstance(sources, list):
        return []  # check() reports the array error.
    problems = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        family, identifier, url = source.get("source"), source.get("id"), source.get("url")
        if family not in {"arxiv", "hf-daily", "hf-search", "web"}:
            problems.append(f"source [{source.get('n')}]: invalid source family {family!r}")
            continue
        prefix = "https://arxiv.org/abs/" if family == "arxiv" else "https://huggingface.co/papers/"
        if family != "web" and (not isinstance(identifier, str) or url != prefix + identifier or
                                 (family == "arxiv" and re.search(r"v\d+$", identifier))):
            problems.append(f"source [{source.get('n')}]: source={family!r}, id={identifier!r}, url={url!r} do not match. "
                            "Consult the original retrieval tool in researcher notes. "
                            "arxiv is ONLY for arxiv_search with an unversioned paper id and canonical arxiv.org/abs URL; "
                            "hf-* is ONLY for its Hugging Face tool. A web_search/web_fetch result must be web. "
                            "Do NOT relabel all web sources as arxiv or fabricate IDs/URLs to pass this check. "
                            "Repair individual JSON entries using execute with Python json.load/json.dump, "
                            "not ambiguous edit_file replacements of repeated strings.")
    return problems


def main(argv):
    strict_metadata = "--metadata" in argv
    argv = [arg for arg in argv if arg != "--metadata"]
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources) + check_structure(report)
    if strict_metadata:
        problems += check_source_metadata(sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
