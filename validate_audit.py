"""Require three supported spot-checks of current claims and sources, inside the sandbox."""
import json
import re
import sys

REPORT = '/tmp/work/report/report.md'
SOURCES = '/tmp/work/research/sources.json'
AUDIT = '/tmp/work/research/citation-audit.json'

def _plain(text):
    return ' '.join(re.sub(r'\[\d+(?:\s*[-,]\s*\d+)*\]', '', text).split())

def validate_audit(report, sources, audit):
    if not isinstance(audit, list):
        return ['citation audit must be a JSON array of actual checker verdicts']
    urls = {s.get('url') for s in sources if isinstance(s, dict)}
    by_n = {s.get('n'): s.get('url') for s in sources if isinstance(s, dict)}
    raw_body = report.split('## References', 1)[0]
    body = _plain(report.split('## References', 1)[0])
    supported = set()
    problems = []
    for entry in audit:
        if not isinstance(entry, dict):
            problems.append('audit entry must be an object')
            continue
        url, claim = entry.get('url'), entry.get('claim')
        if url not in urls:
            problems.append(f'audit URL is absent from finalized sources: {url}')
        elif not isinstance(claim, str) or not _plain(claim) or _plain(claim) not in body:
            problems.append(f'audit claim is absent from current report: {claim}')
        elif not _cites_url(raw_body, claim, url, by_n):
            problems.append(f'audit URL does not match the citation attached to this claim: {claim} ({url})')
        elif entry.get('verdict') != 'SUPPORTED':
            problems.append(f'recheck or remove unresolved claim: {claim} ({entry.get("verdict")})')
        elif not isinstance(entry.get('evidence'), str) or len(entry['evidence'].strip()) < 20:
            problems.append(f'audit needs actual retrieved supporting evidence: {url}')
        else:
            supported.add(url)
    if len(supported) < 3:
        problems.append('need SUPPORTED checker verdicts for current claims from at least three different finalized source URLs')
    return problems

def _cites_url(body, claim, url, by_n):
    # Finalizer may renumber after the checker ran. Inspect citations on the actual current sentence.
    tokens = re.findall(r'\w+|[^\w\s]', _plain(claim))
    pattern = r'(?:\s|\[\d+\])*'.join(re.escape(token) for token in tokens)
    had_citations = False
    for match in re.finditer(pattern, body):
        trailing = re.match(r'\s*((?:\[\d+\]\s*)+)', body[match.end():])
        cited_text = match.group() + (trailing.group(1) if trailing else '')
        numbers = [int(n) for n in re.findall(r'\[(\d+)\]', cited_text)]
        if numbers:
            had_citations = True
            if url in {by_n.get(n) for n in numbers}:
                return True
    if had_citations:
        return False
    # A paragraph-ending reference can support an introductory sentence in that paragraph.
    for paragraph in re.split(r'\n\s*\n', body):
        if _plain(claim) in _plain(paragraph):
            numbers = [int(n) for n in re.findall(r'\[(\d+)\]', paragraph)]
            if url in {by_n.get(n) for n in numbers}:
                return True
    return False

def main():
    try:
        with open(REPORT, encoding='utf-8') as stream:
            report = stream.read()
        with open(SOURCES, encoding='utf-8') as stream:
            sources = json.load(stream)
        with open(AUDIT, encoding='utf-8') as stream:
            audit = json.load(stream)
        problems = validate_audit(report, sources, audit)
        if problems:
            print('\n'.join(problems))
            return 1
        print('AUDIT OK: three current claims have supporting retrieved evidence')
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f'cannot validate citation audit: {exc}')
        return 1

if __name__ == '__main__':
    sys.exit(main())
