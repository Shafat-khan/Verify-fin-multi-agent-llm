"""Offline tests for scripts/check_text_on_real_claims.py (the analysts are faked)."""

import importlib.util
import json
import pathlib

from verifyfin.schemas.claim import AgentRole, ClaimCategory, ClaimObject, SourceReference

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "check_text_on_real_claims.py"
spec = importlib.util.spec_from_file_location("check_text_on_real_claims", SCRIPT)
ct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ct)


def claim(text, excerpt, cat=ClaimCategory.NUMERICAL):
    return ClaimObject(claim_text=text, claim_category=cat, produced_by=AgentRole.FUNDAMENTAL,
                       source_reference=SourceReference(source_type="edgar_xbrl", document_id="x", excerpt=excerpt),
                       confidence_score=0.95, ticker="AAPL", time_period="2026-Q2")


EX = "Revenues: 116017000000 USD (period 2026-03-29 to 2026-06-27)"


class TestEvaluateClaims:
    def test_counts_and_mismatch_detail(self):
        claims = [claim("Total revenue was $116,017,000,000 for the period ended June 27, 2026.", EX),
                  claim("Total revenue was $161,017,000,000 for the period ended June 27, 2026.", EX),
                  claim("Revenue grew strongly.", EX),
                  claim("Some claim without an excerpt.", None)]
        rows, counts = ct.evaluate_claims(claims, "AAPL-2026-Q2")
        assert counts == {"ok": 1, "mismatch": 1, "not_stated": 1, "no_excerpt": 1}
        assert [r["result"] for r in rows] == ["ok", "mismatch", "not_stated"]
        assert "161,017,000,000" in ct.format_summary(rows, counts)

    def test_no_mismatch_message(self):
        rows, counts = ct.evaluate_claims([claim("Total revenue was $116,017,000,000.", EX)], "S")
        assert "No mismatches" in ct.format_summary(rows, counts)


class TestMain:
    def test_runs_over_scenarios_and_survives_a_failing_one(self, tmp_path, capsys):
        results = tmp_path / "evaluation_results_X.json"
        results.write_text(json.dumps([dict(ticker="AAPL", time_period="AAPL-2026-Q2", period_end="2026-06-27"),
                                       dict(ticker="MSFT", time_period="MSFT-2026-Q1", period_end="2026-03-31")]))

        def factory():
            def run(ticker, label, period_end):
                if ticker == "MSFT":
                    raise RuntimeError("boom")
                return [claim("Total revenue was $116,017,000,000 for the period ended June 27, 2026.", EX)]
            return run

        out = tmp_path / "out.json"
        ct.main(["--results", str(results), "--out", str(out)], analysts_factory=factory)
        saved = json.load(open(out))
        assert saved["counts"]["ok"] == 1 and "skipped" in capsys.readouterr().out
