"""Offline contract tests. Run: python -m unittest -v test_lab."""
import json
import tempfile
import unittest
import httpx
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from check_citations import check
import check_citations as validator
from tools import RetryableError, with_retry, arxiv_search, hf_search_papers, web_fetch, _request
from research import slugify, summarize, save_outputs
import research
from agents import REPORT_PATH, SOURCES_PATH
import agents


class LabTests(unittest.TestCase):
    def setUp(self):
        self.sources = [{"n": 1, "url": "https://example.org/a", "title": "A", "date": "2025-01-01", "source": "web", "id": "a"}]
        self.report = "# Survey\nClaim [1].\n## References\n[1] A. web. https://example.org/a (2025-01-01)\n"

    def test_valid_citations(self):
        self.assertEqual(check(self.report, self.sources), [])

    def test_citation_repair_preserves_prose_and_rejects_ambiguous_anchor(self):
        from repair_citations import apply_plan
        report='# Survey\nKnown claim [1]. Other claim [2].\n\n## References\nold refs\n'
        sources=[dict(self.sources[0]),dict(self.sources[0],n=2,url='https://example.org/b')]
        plan={'changes':[{'anchor':'Known claim','urls':['https://example.org/b']} ]}
        repaired, merged=apply_plan(report,sources,plan)
        self.assertIn('Known claim [2]. Other claim [2].',repaired)
        self.assertEqual(merged,sources)
        with self.assertRaises(ValueError):
            apply_plan(report.replace('Other claim','Known claim'),sources,plan)
        with self.assertRaises(ValueError):
            apply_plan(report,sources,{'changes':[{'anchor':'Known claim','urls':['https://unknown.org']}]})

    def test_audit_synchronization_preserves_checker_verdict_and_evidence(self):
        from repair_citations import align_audit
        original=[dict(claim='Exact checked claim [99].',url='https://example.org/a',verdict='SUPPORTED',evidence='Actual fetched evidence.',correction='')]
        result=align_audit('Exact checked claim [2].',original)
        self.assertEqual(result[0]['claim'],'Exact checked claim [2].')
        self.assertEqual({k:v for k,v in result[0].items() if k!='claim'},{k:v for k,v in original[0].items() if k!='claim'})
        self.assertEqual(original[0]['claim'],'Exact checked claim [99].')

    def test_audit_matches_citations_after_sentence_period(self):
        sources=[dict(self.sources[0],n=i,url=f'https://example.org/{i}') for i in range(1,4)]
        claims=['First exact claim.','Second exact claim.','Third exact claim.']
        report='First exact claim. [1] Second exact claim. [2] Third exact claim. [3]'
        audit=[dict(claim=c,url=s['url'],verdict='SUPPORTED',evidence='Actual retrieved supporting evidence.') for c,s in zip(claims,sources)]
        self.assertEqual(research.validate_audit(report,sources,audit),[])
        rotated=[dict(entry,url=sources[(i+1)%3]['url']) for i,entry in enumerate(audit)]
        self.assertTrue(research.validate_audit(report,sources,rotated))

    def test_audit_requires_three_supported_current_claims(self):
        sources=[dict(self.sources[0],n=i,url=f'https://example.org/{i}') for i in range(1,4)]
        report='First supported claim [1]. Second supported claim [2]. Third supported claim [3].'
        audit=[dict(url=s['url'],claim=c,verdict='SUPPORTED',evidence='Retrieved evidence supporting the exact claim.') for s,c in zip(sources,['First supported claim [1].','Second supported claim [2].','Third supported claim [3].'])]
        self.assertEqual(research.validate_audit(report,sources,audit),[])
        self.assertTrue(research.validate_audit(report,sources,[dict(audit[0],verdict='UNSUPPORTED'),*audit[1:]]))
        self.assertTrue(research.validate_audit(report,sources,[dict(audit[0],url='https://unknown.org'),*audit[1:]]))
        self.assertTrue(research.validate_audit(report,sources,[dict(audit[0],claim='Absent claim.'),*audit[1:]]))
        rotated=[dict(entry,url=sources[(i+1)%3]['url']) for i,entry in enumerate(audit)]
        self.assertTrue(research.validate_audit(report,sources,rotated))
        self.assertEqual(research.validate_audit(report,sources,[dict(audit[0],claim='First supported claim [99].'),*audit[1:]]),[])

    def test_structure_rejects_uncited_substantive_paragraph(self):
        body = '# Survey\n## TL;DR\n- Finding [1].\n## Background\nDefinition [1].\n## A\nA [1].\n## B\nB [1].\n## C\nC [1].\n## Trends and open problems\nRecent approaches reduce computational costs through several architectural improvements.\n## References\n[1] A\n'
        self.assertTrue(validator.check_structure(body))

    def test_arxiv_metadata_rejects_versioned_identifier(self):
        self.assertTrue(validator.check_source_metadata([dict(self.sources[0], source='arxiv', id='2501.00001v2', url='https://arxiv.org/abs/2501.00001v2')]))

    def test_report_structure_requires_template_and_background_citation(self):
        body = "# Survey\n## TL;DR\n- Finding [1]\n## Background\nDefinition [1].\n## Theme A\nA [1].\n## Theme B\nB [1].\n## Theme C\nC [1].\n## Trends and open problems\nLimits [1].\n## References\n[1] A. web. https://example.org/a (2025-01-01)\n"
        self.assertEqual(validator.check_structure(body), [])
        self.assertTrue(validator.check_structure(body.replace("Trends and open problems", "Trends and Open Problems")))
        self.assertTrue(validator.check_structure(body.replace("Definition [1]", "Definition")))
        self.assertTrue(validator.check_structure(body.replace("## Theme C\nC [1].\n", "")))

    def test_source_metadata_rejects_mislabelled_web_urls(self):
        self.assertEqual(validator.check_source_metadata(self.sources), [])
        bad = [dict(self.sources[0], source="arxiv")]
        problems = validator.check_source_metadata(bad)
        self.assertTrue(problems)
        self.assertIn("source [1]", problems[0])
        self.assertIn("retrieval tool", problems[0])

    def test_provenance_normalization_uses_actual_tool_records(self):
        from normalize_sources import normalize
        original = [dict(self.sources[0], source="arxiv", title="Invented title")]
        ledger = {"https://example.org/a": [dict(self.sources[0], id="https://example.org/a")]}
        normalized, problems = normalize(original, ledger)
        self.assertEqual(problems, [])
        self.assertEqual(normalized[0]["source"], "web")
        self.assertEqual(normalized[0]["title"], "A")
        self.assertEqual(original[0]["title"], "Invented title")
        self.assertTrue(normalize([dict(original[0], url="https://unknown.org")], ledger)[1])

    def test_tools_record_source_provenance(self):
        from tools import clear_source_ledger, source_ledger
        clear_source_ledger()
        payload = [{"paper": {"id": "2501.00001", "title": "A", "summary": "Evidence", "publishedAt": "2025-01-02"}}]
        response = httpx.Response(200, json=payload, request=httpx.Request("GET", "https://huggingface.co"))
        with patch("tools.httpx.get", return_value=response):
            hf_search_papers.invoke({"query": "test"})
        records = source_ledger()["https://huggingface.co/papers/2501.00001"]
        self.assertEqual(records[0]["source"], "hf-search")
        self.assertEqual(records[0]["date"], "2025-01-02")

    def test_fetch_ledger_ignores_embedded_phantom_urls(self):
        from tools import clear_source_ledger, source_ledger, _remember_web
        clear_source_ledger()
        _remember_web('# Real page\nURL: https://example.org/a\nPublished: 2025-01-01\n\nBody\nTitle: Fake\nURL: https://phantom.org/x\n', 'https://example.org/a')
        self.assertEqual(set(source_ledger()), {'https://example.org/a'})

    def test_reject_missing_unused_duplicate_and_bundled(self):
        for report, sources in [
            (self.report.replace("Claim [1]", "Claim [2]"), self.sources),
            (self.report.replace("Claim [1]", "Claim"), self.sources),
            (self.report, self.sources * 2),
            (self.report.replace("(2025-01-01)", "https://example.org/b"), self.sources),
            (self.report.replace("## References", "## Bibliography"), self.sources),
            (self.report.replace("Claim [1]", "```\n[1]\n```\n[1](https://example.org/a)"), self.sources),
            (self.report.replace("Claim [1].", "    example [1]"), self.sources),
        ]:
            with self.subTest(report=report):
                self.assertTrue(check(report, sources))

    def test_group_citations(self):
        sources = self.sources + [dict(self.sources[0], n=2, url="https://example.org/b")]
        report = self.report.replace("Claim [1]", "Claim [1-2]") + "[2] B. web. https://example.org/b (2025-01-01)\n"
        self.assertEqual(check(report, sources), [])

    @patch("tools.time.sleep")
    def test_retry_after_and_no_final_sleep(self, sleep):
        calls = []
        def fail():
            calls.append(1)
            raise RetryableError("busy", retry_after=2)
        with self.assertRaises(RetryableError):
            with_retry(fail, attempts=3)
        self.assertEqual(len(calls), 3)
        self.assertEqual([x.args[0] for x in sleep.call_args_list], [2, 2])

    @patch("tools.time.sleep")
    def test_nonretryable(self, sleep):
        with self.assertRaises(ValueError):
            with_retry(lambda: (_ for _ in ()).throw(ValueError("bad")))
        sleep.assert_not_called()

    def test_slug_and_metadata(self):
        self.assertEqual(slugify("../../World Model"), "world-model")
        self.assertEqual(slugify("!!!"), "topic")
        self.assertLessEqual(len(slugify("x" * 100)), 60)
        msg = SimpleNamespace(tool_calls=[{"name": "task"}], usage_metadata={"input_tokens": 10, "output_tokens": 4})
        meta = summarize([msg], 1.25, "test")
        self.assertEqual(meta["subagent_calls"], 1)
        self.assertEqual(meta["tokens"], {"input": 10, "output": 4})

    def test_metadata_distinguishes_real_research_and_checker_tasks(self):
        msg = SimpleNamespace(tool_calls=[{"name": "task", "args": {"subagent_type": name}}
                              for name in ["researcher", "researcher", "researcher", "citation-checker"]], usage_metadata={})
        meta = summarize([msg], 0, "test")
        self.assertEqual(meta.get("researcher_calls"), 3)
        self.assertEqual(meta.get("citation_checker_calls"), 1)

    def test_main_repairs_in_same_sandbox_before_saving(self):
        backend = SimpleNamespace(execute=lambda command: SimpleNamespace(exit_code=0, output="OK"))
        context = unittest.mock.MagicMock()
        context.__enter__.return_value = backend
        history = [SimpleNamespace(tool_calls=[], usage_metadata={})]
        with patch("research.make_model", return_value=SimpleNamespace()), \
             patch("research.open_sandbox", return_value=context) as opened, \
             patch("research.upload"), patch("research.build_lead_agent", return_value=object()), \
             patch("research._run_agent", return_value=history) as run, \
             patch("research._validate_sandbox", side_effect=[RuntimeError("source mismatch"), "OK: 3 sources"]), \
             patch("research.save_outputs", return_value=Path("reports/test.md")) as save:
            self.assertEqual(research.main("test"), 0)
        opened.assert_called_once()
        self.assertEqual(run.call_count, 2)
        repair_messages = run.call_args_list[1].args[1]
        self.assertIs(repair_messages[0], history[0])
        self.assertIn("source mismatch", repair_messages[-1]["content"])
        save.assert_called_once()

    def test_main_caps_repairs_and_does_not_save_failure(self):
        backend = SimpleNamespace(execute=lambda command: SimpleNamespace(exit_code=0, output="OK"))
        context = unittest.mock.MagicMock()
        context.__enter__.return_value = backend
        with patch("research.make_model", return_value=SimpleNamespace()), \
             patch("research.open_sandbox", return_value=context), patch("research.upload"), \
             patch("research.build_lead_agent", return_value=object()), \
             patch("research._run_agent", return_value=[]) as run, \
             patch("research._validate_sandbox", side_effect=RuntimeError("source mismatch")), \
             patch("research.save_outputs") as save:
            self.assertEqual(research.main("test"), 1)
        self.assertEqual(run.call_count, 3)
        save.assert_not_called()
        context.__exit__.assert_called_once()

    def test_invalid_download_writes_nothing(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            with patch("research.download", return_value={REPORT_PATH: b"", SOURCES_PATH: b"[]"}):
                with self.assertRaises(RuntimeError):
                    save_outputs(None, "test", [], 0, "test", Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_valid_download_preserves_bytes(self):
        files = {REPORT_PATH: self.report.encode(), SOURCES_PATH: json.dumps(self.sources).encode()}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            with patch("research.download", return_value=files):
                path = save_outputs(None, "test", [], 0, "test", Path(directory))
            self.assertEqual(path.read_bytes(), files[REPORT_PATH])
            self.assertEqual(path.with_suffix(".sources.json").read_bytes(), files[SOURCES_PATH])

    def test_empty_arxiv_does_not_call_network(self):
        with patch("tools.httpx.get") as get:
            self.assertEqual(arxiv_search.invoke({"query": '" : AND OR'}), "NO RESULTS")
            get.assert_not_called()

    def test_tools_do_not_raise(self):
        with patch("tools.httpx.get", side_effect=ValueError("bad")):
            self.assertTrue(hf_search_papers.invoke({"query": "test"}).startswith("ERROR:"))
        with patch("tools.httpx.post", side_effect=ValueError("bad")):
            self.assertTrue(web_fetch.invoke({"url": "https://example.org"}).startswith("ERROR:"))

    def test_arxiv_normalizes_atom(self):
        xml = '<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2501.00001v2</id><published>2025-01-02T00:00:00Z</published><title>A\n paper</title><summary>Evidence</summary></entry></feed>'
        with patch("tools._request", return_value=SimpleNamespace(text=xml)), patch("tools.time.sleep"):
            result = json.loads(arxiv_search.invoke({"query": 'all:"world" AND model'}))
        self.assertEqual(result[0]["url"], "https://arxiv.org/abs/2501.00001")
        self.assertEqual(result[0]["title"], "A paper")

    def test_http_retry_status_and_header(self):
        response = httpx.Response(429, headers={"Retry-After": "3"}, request=httpx.Request("GET", "https://example.org"))
        with patch("tools.httpx.get", return_value=response):
            with self.assertRaises(RetryableError) as caught:
                _request("GET", "https://example.org")
        self.assertEqual(caught.exception.retry_after, 3)

    def test_mcp_rate_limit_http_200_then_success(self):
        limited = {"result": {"_meta": {"rateLimited": True}, "content": [{"type": "text", "text": "Busy"}]}}
        success = {"result": {"content": [{"type": "text", "text": "Evidence"}]}}
        def response(payload):
            return httpx.Response(200, text="event: message\ndata: " + json.dumps(payload), request=httpx.Request("POST", "https://mcp.exa.ai/mcp"))
        with patch("tools.httpx.post", side_effect=[response(limited), response(success)]), patch("tools.time.sleep") as sleep:
            self.assertEqual(web_fetch.invoke({"url": "https://example.org"}), "Evidence")
        sleep.assert_called_once()

    def test_api_key_redacted_in_error(self):
        with patch.dict("os.environ", {"EXA_API_KEY": "fake-test-secret"}), patch("tools.httpx.post", side_effect=ValueError("https://mcp.exa.ai/mcp?exaApiKey=fake-test-secret")):
            output = web_fetch.invoke({"url": "https://example.org"})
        self.assertNotIn("fake-test-secret", output)
        self.assertIn("REDACTED", output)

    def test_mcp_false_rate_limit_flag_is_not_an_error(self):
        payload = {"result": {"_meta": {"rateLimited": False}, "content": [{"type": "text", "text": "Evidence"}]}}
        response = httpx.Response(200, json=payload, request=httpx.Request("POST", "https://mcp.exa.ai/mcp"))
        with patch("tools.httpx.post", return_value=response) as post, patch("tools.time.sleep"):
            self.assertEqual(web_fetch.invoke({"url": "https://example.org"}), "Evidence")
            self.assertEqual(post.call_count, 1)

    def test_model_transport_error_is_retried_without_restarting_graph(self):
        retry = next((m for m in agents._limits(4, 8) if type(m).__name__ == "ModelRetryMiddleware"), None)
        self.assertIsNotNone(retry)
        calls = []
        def handler(request):
            calls.append(1)
            if len(calls) == 1:
                raise httpx.ConnectError("temporary")
            return "response"
        with patch("time.sleep"):
            self.assertEqual(retry.wrap_model_call(SimpleNamespace(), handler), "response")
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()

