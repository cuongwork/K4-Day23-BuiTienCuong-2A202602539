"""Apply reviewed citation-only mappings in a sandbox; never rewrite report prose or call an LLM."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

REPORT = '/tmp/work/report/report.md'
SOURCES = '/tmp/work/research/sources.json'
PLAN = '/tmp/work/research/citation-repair-plan.json'
AUDIT = '/tmp/work/research/citation-audit.json'
ROOT = Path(__file__).parent

def apply_plan(report, sources, plan):
    """Change citation groups after unique prose anchors, using verified source URLs."""
    merged = [dict(source) for source in sources]
    by_url = {source['url']: source['n'] for source in merged}
    next_n = max((source['n'] for source in merged), default=0) + 1
    for record in plan.get('append_sources', []):
        if record['url'] not in by_url:
            merged.append(dict(record, n=next_n))
            by_url[record['url']] = next_n
            next_n += 1
    split = re.search(r'(?m)^## References\s*$', report)
    body = report[:split.start()] if split else report
    tail = report[split.start():] if split else ''
    original = body
    for change in plan['changes']:
        anchor = change['anchor']
        if not anchor or body.count(anchor) != 1:
            raise ValueError(f'citation anchor must occur exactly once: {anchor}')
        if not change['urls'] or any(url not in by_url for url in change['urls']):
            raise ValueError(f'unknown or empty source mapping: {anchor}')
        pattern = re.escape(anchor) + r'(?P<context>[^\n]*?)(?P<cites>(?:\[\d+\])+)'
        match = re.search(pattern, body)
        if match is None:
            raise ValueError(f'no citation group after anchor: {anchor}')
        replacement = ''.join(f'[{by_url[url]}]' for url in dict.fromkeys(change['urls']))
        body = body[:match.start('cites')] + replacement + body[match.end('cites'):]
    strip = lambda text: re.sub(r'\[\d+\]', '', text)
    if strip(body) != strip(original):
        raise ValueError('citation repair must preserve all report prose')
    return body + tail, merged

def sandbox_main():
    try:
        report = Path(REPORT).read_text(encoding='utf-8')
        sources = json.loads(Path(SOURCES).read_text(encoding='utf-8'))
        plan = json.loads(Path(PLAN).read_text(encoding='utf-8'))
        repaired, merged = apply_plan(report, sources, plan)
        Path(REPORT).write_text(repaired, encoding='utf-8')
        Path(SOURCES).write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'REPAIRED: {len(plan["changes"])} citation groups; report prose unchanged')
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'cannot repair citations: {exc}')
        return 1

def align_audit(report, audit):
    """Copy current citation numbers into prior verified quotations; preserve all verdicts and evidence."""
    body = report.split('## References', 1)[0]
    plain = lambda text: ' '.join(re.sub(r'\[\d+\]', '', text).split())
    aligned = []
    for entry in audit:
        tokens = re.findall(r'\w+|[^\w\s]', plain(entry['claim']))
        pattern = r'(?:\s|\[\d+\])*'.join(re.escape(token) for token in tokens)
        matches = list(re.finditer(pattern, body))
        if len(matches) != 1:
            raise ValueError('audit quotation must match exactly one current sentence')
        match = matches[0]
        trailing = re.match(r'\s*((?:\[\d+\]\s*)+)', body[match.end():])
        claim = match.group() + (trailing.group(1).rstrip() if trailing else '')
        if plain(claim) != plain(entry['claim']):
            raise ValueError('audit synchronization must preserve the verified claim')
        aligned.append(dict(entry, claim=claim))
    return aligned

def audit_main():
    try:
        report = Path(REPORT).read_text(encoding='utf-8')
        audit = json.loads(Path(AUDIT).read_text(encoding='utf-8'))
        Path(AUDIT).write_text(json.dumps(align_audit(report, audit), ensure_ascii=False, indent=2), encoding='utf-8')
        print('AUDIT SYNCHRONIZED: current reference numbers; original verdicts and evidence preserved')
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'cannot synchronize audit: {exc}')
        return 1

def run(plan_path, original_stem):
    # Third-party dependencies and network tools stay on the host, not in the sandbox.
    from concurrent.futures import ThreadPoolExecutor
    from check_citations import check, check_structure, check_source_metadata
    from research import _execute_ok
    from sandbox import open_sandbox, upload, download
    from tools import clear_source_ledger, source_ledger, web_fetch
    from validate_audit import validate_audit

    started = time.monotonic()
    stem = Path(original_stem)
    original_report = stem.with_suffix('.md').read_bytes()
    original_sources = stem.with_suffix('.sources.json').read_bytes()
    meta = json.loads(stem.with_suffix('.meta.json').read_bytes())
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    clear_source_ledger()
    urls = sorted(set(plan.get('new_urls', [])) | {url for change in plan['changes'] for url in change['urls']})
    def retrieve(url):
        text = web_fetch.invoke({'url': url})
        if text.startswith(('ERROR:', 'NO RESULTS')):
            raise RuntimeError(f'cannot retrieve repair evidence: {url}')
        return {'url': url, 'text': text}
    evidence = list(ThreadPoolExecutor(4).map(retrieve, urls))
    ledger = source_ledger()
    plan['append_sources'] = []
    for url in plan.get('new_urls', []):
        if not ledger.get(url):
            raise RuntimeError(f'no successful retrieval metadata for added source: {url}')
        plan['append_sources'].append(ledger[url][0])
    plan_bytes = json.dumps(plan, ensure_ascii=False, indent=2).encode()
    audit_bytes = json.dumps(meta['citation_audit'], ensure_ascii=False, indent=2).encode()
    with open_sandbox() as backend:
        _execute_ok(backend, 'mkdir -p /tmp/work/research /tmp/work/report')
        upload(backend, {REPORT: original_report, SOURCES: original_sources, PLAN: plan_bytes,
                         AUDIT: audit_bytes,
                         '/tmp/work/research/repair-evidence.json': json.dumps(evidence, ensure_ascii=False).encode(),
                         '/tmp/work/repair_citations.py': Path(__file__).read_bytes(),
                         '/tmp/work/research/check_citations.py': (ROOT/'check_citations.py').read_bytes(),
                         '/tmp/work/research/validate_audit.py': (ROOT/'validate_audit.py').read_bytes(),
                         '/tmp/work/research/finalize_citations.py': (ROOT/'finalize_citations.py').read_bytes()})
        for command in ['python3 /tmp/work/repair_citations.py --sandbox',
                        'python3 /tmp/work/research/finalize_citations.py',
                        'python3 /tmp/work/repair_citations.py --align-audit',
                        'python3 /tmp/work/research/check_citations.py --metadata',
                        'python3 /tmp/work/research/validate_audit.py']:
            print(_execute_ok(backend, command).strip(), flush=True)
        files = download(backend, [REPORT, SOURCES, AUDIT])
    body = files[REPORT].decode('utf-8')
    sources = json.loads(files[SOURCES])
    audit = json.loads(files[AUDIT])
    problems = check(body, sources)+check_structure(body)+check_source_metadata(sources)+validate_audit(body, sources, audit)
    families = sorted({source['source'] for source in sources})
    if problems or len(families) < 3:
        raise RuntimeError(f'invalid repaired output: {problems}; families={families}')
    # Preserve the real original agent statistics; this pass has zero model calls.
    meta.update(n_sources=len(sources), source_families=families, citation_audit=audit)
    meta['citation_repair'] = {'method': 'deterministic source-URL mapping in sandbox', 'llm_calls': 0,
        'retrieval_calls': len(urls), 'added_sources': len(plan['append_sources']),
        'elapsed_s': round(time.monotonic()-started, 1),
        'original_report_sha256': hashlib.sha256(original_report).hexdigest(),
        'repaired_report_sha256': hashlib.sha256(files[REPORT]).hexdigest(),
        'plan_sha256': hashlib.sha256(plan_bytes).hexdigest()}
    meta['citation_repair']['original_audit_sha256'] = hashlib.sha256(audit_bytes).hexdigest()
    meta['citation_repair']['audit_citation_numbers_synchronized'] = True
    destination = ROOT/'reports'
    destination.mkdir(exist_ok=True)
    outputs = {stem.name+'.md': files[REPORT], stem.name+'.sources.json': files[SOURCES],
               stem.name+'.meta.json': (json.dumps(meta, ensure_ascii=False, indent=2)+'\n').encode()}
    staged = []
    try:
        for name, content in outputs.items():
            with tempfile.NamedTemporaryFile(dir=destination, prefix='.stage-', delete=False) as stream:
                stream.write(content)
                staged.append((Path(stream.name), destination/name))
        for temporary, target in staged:
            os.replace(temporary, target)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    print(f'Saved: {destination/(stem.name+".md")}', flush=True)
    return 0

if __name__ == '__main__':
    if sys.argv[1:] == ['--sandbox']:
        sys.exit(sandbox_main())
    if sys.argv[1:] == ['--align-audit']:
        sys.exit(audit_main())
    if len(sys.argv) != 3:
        print('Usage: python repair_citations.py <plan.json> <original-report-stem>')
        sys.exit(2)
    try:
        sys.exit(run(sys.argv[1], sys.argv[2]))
    except Exception as exc:
        from tools import redact
        print(f'FAILED: {type(exc).__name__}: {redact(exc)}', file=sys.stderr)
        sys.exit(1)
