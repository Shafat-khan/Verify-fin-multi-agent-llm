"""End-to-end test of scripts/run_fault_injection.py main() with injected fake dependencies (no network, no Chroma, no LLM)."""

import csv
import importlib.util
import json
import pathlib
import re

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
TP = "AAPL-2026-Q2"
CHUNKS = [("Total net sales 109,417 for the three months ended June 27, 2026", "edgar_10q"),
          ("Net income 29,789 for the three months ended June 27, 2026", "edgar_10q"),
          ("FEDFUNDS observation: 3.63 on 2026-07-01", "fred"), ("FEDFUNDS observation: 3.80 on 2026-08-01", "fred"),
          ("UNRATE observation: 4.1 on 2026-07-01", "fred"), ("UNRATE observation: 4.1 on 2026-08-01", "fred")]


def _tok(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


class FakeStore:
    def query(self, q, ticker=None, top_k=5, where_document_contains=None):
        qs, out = _tok(q), []
        for t, st in CHUNKS:
            if where_document_contains and where_document_contains not in t:
                continue
            ts = _tok(t)
            out.append({"text": t, "metadata": {"ticker": "AAPL", "source_type": st}, "similarity_score": round(0.45 + 0.4 * len(qs & ts) / len(qs | ts), 4)})
        return sorted(out, key=lambda r: -r["similarity_score"])[:top_k]

    def count(self):
        return len(CHUNKS)


class FakeDebate:
    def debate_claim(self, claim):
        claim.debate_resolution = "rejected"
        return claim


class FakeDeps:
    def __init__(self, scen, **kw):
        self.scen = scen
        self.store = FakeStore()

    def get_store(self, tp):
        return self.store

    def validator_factory(self, tp):
        from verifyfin.evaluation.fault_validators import make_validator
        truth = {"revenue": {"value": 109417000000}, "net_income": {"value": 29789000000}}
        return make_validator(ticker="AAPL", period_end="2026-06-27", store=self.store, xbrl_truth=lambda t, pe: truth,
                              known_period_ends=["2026-06-27"])

    def debate_agent(self):
        return FakeDebate()


def _load():
    spec = importlib.util.spec_from_file_location("run_fault_injection", SCRIPTS / "run_fault_injection.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def files(tmp_path):
    res = [{"ticker": "AAPL", "time_period": TP, "period_end": "2026-06-27", "condition": "verify_fin"}]
    json.dump(res, open(tmp_path / "res.json", "w"))
    rows = [dict(claim_id="c1", ticker="AAPL", time_period=TP, claim_category="numerical", claim_text="Apple reported total revenue of $109,417,000,000 for the period ending June 27, 2026.", human_label="SUPPORTED"),
            dict(claim_id="c2", ticker="AAPL", time_period=TP, claim_category="macro", claim_text="On 2026-07-01, the Federal Funds Rate was 3.63.", human_label="SUPPORTED"),
            dict(claim_id="c3", ticker="AAPL", time_period=TP, claim_category="macro", claim_text="On 2026-07-01, the U.S. unemployment rate was 4.1.", human_label="SUPPORTED"),
            dict(claim_id="c4", ticker="AAPL", time_period=TP, claim_category="sentiment", claim_text="Headlines are negative.", human_label="UNSUPPORTED")]
    with open(tmp_path / "ws.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    key = {r["claim_id"]: dict(verdict="grounded", debate_resolution=None, grounding_score=0.86, is_usable=True, outcome_bucket="GROUNDED") for r in rows}
    json.dump(key, open(tmp_path / "key.json", "w"))
    return tmp_path


def _argv(d, *extra):
    return ["--results", str(d / "res.json"), "--worksheet", str(d / "ws.csv"), "--answer-key", str(d / "key.json"), "--out-dir", str(d / "out"), *extra]


class TestCli:
    def test_dry_run_touches_no_knowledge_base(self, files):
        m = _load()
        called = []
        m.main(_argv(files, "--dry-run"), deps_factory=lambda scen, **k: called.append(1) or FakeDeps(scen))
        assert not called

    def test_full_run_writes_report_csv_and_meta(self, files):
        m = _load()
        m.main(_argv(files, "--n-boot", "100", "--debate-budget", "4"), deps_factory=FakeDeps)
        out = files / "out"
        names = sorted(p.name for p in out.iterdir())
        assert any(n.startswith("fault_injection_report_") for n in names)
        assert any(n.startswith("fault_injection_rows_") for n in names)
        meta = json.load(open(next(out.glob("fault_injection_meta_*.json"))))
        assert meta["validated"] is True and meta["n_bases"] == 3
        report = next(out.glob("fault_injection_report_*.txt")).read_text()
        for needle in ["FAULT-INJECTION EXPERIMENT", "PRIMARY RESULT", "DEBATE-STAGE SAMPLE", "ABLATION"]:
            assert needle in report

    def test_flat_unemployment_series_does_not_yield_a_date_shift_corruption(self, files):
        """UNRATE is 4.1 in both July and August in the fake KB, so a month shift would create a TRUE claim and must be dropped."""
        m = _load()
        m.main(_argv(files, "--n-boot", "50", "--configs", "full|all"), deps_factory=FakeDeps)
        rows = list(csv.DictReader(open(next((files / "out").glob("fault_injection_rows_*.csv")), newline="")))
        shifted = [r for r in rows if r["kind"] == "corrupted" and r["operator"] == "date_shift_month" and r["base_id"] == "c3"]
        assert not shifted
        assert [r for r in rows if r["kind"] == "corrupted" and r["operator"] == "date_shift_month" and r["base_id"] == "c2"]  # FFR moves 3.63 -> 3.80

    def test_unknown_variant_is_rejected(self, files):
        m = _load()
        with pytest.raises(SystemExit):
            m.main(_argv(files, "--configs", "nonsense|all"), deps_factory=FakeDeps)

    def test_primary_config_is_always_included(self, files):
        m = _load()
        m.main(_argv(files, "--n-boot", "50", "--configs", "strict_date|all"), deps_factory=FakeDeps)
        meta = json.load(open(next((files / "out").glob("fault_injection_meta_*.json"))))
        assert "full|all" in meta["configs"] and "strict_date|all" in meta["configs"]

    def test_quick_mode_limits_corrupted_claims(self, files):
        m = _load()
        m.main(_argv(files, "--n-boot", "50", "--quick", "1", "--configs", "full|all"), deps_factory=FakeDeps)
        rows = list(csv.DictReader(open(next((files / "out").glob("fault_injection_rows_*.csv")), newline="")))
        assert len({r["base_id"] for r in rows if r["kind"] == "corrupted"}) == 1

    def test_v2_flag_adds_text_check_configs_and_debates_them(self, files):
        m = _load()
        m.main(_argv(files, "--n-boot", "50", "--configs", "full|all", "--v2", "--debate-budget", "3",
                     "--debate-config", "text_check|all"), deps_factory=FakeDeps)
        meta = json.load(open(next((files / "out").glob("fault_injection_meta_*.json"))))
        for c in ("full|all", "text_check|all", "text_check_strict|all", "text_check_block|all"):
            assert c in meta["configs"]
        report = next((files / "out").glob("fault_injection_report_*.txt")).read_text()
        assert "DEBATE-STAGE SAMPLE (configuration 'text_check|all')" in report

    def test_debate_config_is_added_even_without_v2(self, files):
        m = _load()
        m.main(_argv(files, "--n-boot", "50", "--configs", "full|all", "--debate-budget", "2", "--debate-config", "text_check|all"),
               deps_factory=FakeDeps)
        meta = json.load(open(next((files / "out").glob("fault_injection_meta_*.json"))))
        assert "text_check|all" in meta["configs"]


class TestV3Flag:
    def test_v3_flag_adds_the_alignment_configs(self, files):
        m = _load()
        m.main(_argv(files, "--n-boot", "50", "--configs", "full|all", "--v3"), deps_factory=FakeDeps)
        meta = json.load(open(next((files / "out").glob("fault_injection_meta_*.json"))))
        for c in ("full|all", "text_check|all", "v3_date|all", "v3_label|all", "v3_period|all", "v3_aligned|all", "v3_exact|all"):
            assert c in meta["configs"]

    def test_dry_run_counts_v3_configs(self, files, capsys):
        m = _load()
        m.main(_argv(files, "--configs", "full|all", "--v3", "--dry-run"), deps_factory=FakeDeps)
        assert "7 verifier configurations" in capsys.readouterr().out   # full + text_check + 5 v3 variants
