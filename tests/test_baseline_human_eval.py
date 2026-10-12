"""Tests for scripts/baseline_human_eval.py (offline: the knowledge-base context builder is injected)."""

import csv
import importlib.util
import json
import pathlib

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "baseline_human_eval.py"
spec = importlib.util.spec_from_file_location("baseline_human_eval", SCRIPT)
bhe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bhe)


def rec(i, tp="AAPL-2026-Q2", text=None):
    return dict(claim_id=f"AAPL_{tp}_claim{i}", ticker="AAPL", time_period=tp, claim_category="free_text",
                claim_text=text or f"Claim {i}", retrieved_evidence="one chunk", verdict="challenged",
                grounding_score=0.5, is_usable=False, debate_resolution=None)


class TestPureHelpers:
    def test_duplicate_ids_get_suffixes(self):
        out = bhe.unique_ids([dict(claim_id="a"), dict(claim_id="a"), dict(claim_id="b")])
        assert [r["claim_id"] for r in out] == ["a", "a__2", "b"]

    def test_sample_takes_all_when_few(self):
        assert len(bhe.sample_across_scenarios([rec(i) for i in range(5)], 60)) == 5

    def test_sample_is_spread_across_scenarios_and_reproducible(self):
        recs = [rec(i, "S1") for i in range(30)] + [rec(i + 100, "S2") for i in range(3)] + [rec(i + 200, "S3") for i in range(3)]
        s = bhe.sample_across_scenarios(recs, 9, seed=1)
        assert len(s) == 9 and {r["time_period"] for r in s} == {"S1", "S2", "S3"}
        assert [r["claim_id"] for r in s] == [r["claim_id"] for r in bhe.sample_across_scenarios(recs, 9, seed=1)]

    def test_worksheet_is_blind_and_uses_full_context(self):
        rows = bhe.worksheet_rows([rec(1)], {"AAPL-2026-Q2": "[1] (fred) all passages"})
        assert rows[0]["retrieved_evidence"] == "[1] (fred) all passages"
        assert set(rows[0]) == set(bhe.FIELDS) and "verdict" not in rows[0] and rows[0]["human_label"] == ""

    def test_error_rate_counts_everything_but_supported(self):
        r = bhe.error_rate(["SUPPORTED"] * 8 + ["UNSUPPORTED", "CONTRADICTED"])
        assert r["n"] == 10 and r["not_supported"] == 2 and r["rate"] == pytest.approx(0.2) and r["lo"] < 0.2 < r["hi"]

    def test_end_to_end_rates(self):
        main = [dict(claim_id=f"c{i}", human_label=l) for i, l in enumerate(["SUPPORTED", "SUPPORTED", "UNSUPPORTED", "SUPPORTED"])]
        key = {"c0": {"is_usable": True}, "c1": {"is_usable": True}, "c2": {"is_usable": False}, "c3": {"is_usable": True}}
        res = bhe.end_to_end_rates(main, key, [dict(human_label="SUPPORTED"), dict(human_label="unsupported")])
        assert res["unverified_multiagent"]["not_supported"] == 1 and res["unverified_multiagent"]["n"] == 4
        assert res["verify_fin"]["not_supported"] == 0 and res["verify_fin"]["n"] == 3      # the bad claim was filtered
        assert res["single_agent_rag"]["rate"] == 0.5
        assert res["removed_by_verify_fin"] == {"n": 1, "truly_unsupported": 1}
        assert "ERROR LEFT IN THE REPORT" in bhe.format_scores(res)


class TestEndToEnd:
    def _setup(self, tmp_path):
        results = tmp_path / "evaluation_results_X.json"
        results.write_text(json.dumps([dict(ticker="AAPL", time_period="AAPL-2026-Q2", period_end="2026-06-27")]))
        claims = tmp_path / "rag_claims_X.jsonl"
        claims.write_text("\n".join(json.dumps(rec(i)) for i in range(12)))
        return results, claims

    def test_export_then_score(self, tmp_path, capsys):
        results, claims = self._setup(tmp_path)
        bhe.main(["export", "--claims", str(claims), "--results", str(results), "--n", "8", "--out-dir", str(tmp_path)],
                 context_builder=lambda scen, tps: {tp: "[1] (edgar_10q) evidence" for tp in tps})
        ws = tmp_path / "rag_worksheet.csv"
        rows = list(csv.DictReader(open(ws)))
        assert len(rows) == 8 and all(r["retrieved_evidence"] == "[1] (edgar_10q) evidence" for r in rows)
        key = json.load(open(tmp_path / "rag_answer_key.json"))
        assert set(key) == {r["claim_id"] for r in rows}

        # scoring refuses to run on unlabelled data
        main_ws = tmp_path / "main.csv"
        main_ws.write_text("claim_id,human_label\nc0,SUPPORTED\nc1,UNSUPPORTED\n")
        main_key = tmp_path / "main_key.json"
        main_key.write_text(json.dumps({"c0": {"is_usable": True}, "c1": {"is_usable": False}}))
        with pytest.raises(SystemExit):
            bhe.main(["score", "--out-dir", str(tmp_path), "--main-worksheet", str(main_ws), "--main-key", str(main_key)])

        for r in rows:
            r["human_label"] = "SUPPORTED"
        rows[0]["human_label"] = "UNSUPPORTED"
        with open(ws, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=bhe.FIELDS)
            w.writeheader()
            w.writerows(rows)
        bhe.main(["score", "--out-dir", str(tmp_path), "--main-worksheet", str(main_ws), "--main-key", str(main_key)])
        out = capsys.readouterr().out
        assert "single_agent_rag" in out and "12.5%" in out
