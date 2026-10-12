"""
Tests for scripts/run_evaluation.py orchestration, with every heavy dependency stubbed: the resume (--fill-missing)
logic, the opt-in Single-Agent + RAG condition, knowledge-base reuse, and the file-naming guard that stops the RAG
baseline's claims from being mistaken for VERIFY-FIN's.
"""

import importlib.util
import json
import pathlib
import sys
import types
from unittest.mock import MagicMock

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"


def _load(tmp):
    stubs = {
        "verifyfin.baselines.single_agent_baseline": {"SingleAgentBaseline": MagicMock()},
        "verifyfin.baselines.single_agent_rag_baseline": {"SingleAgentRagBaseline": MagicMock()},
        "verifyfin.baselines.unverified_multiagent_baseline": {"UnverifiedMultiAgentBaseline": MagicMock()},
        "verifyfin.data_pipeline.chunker": {"Chunker": MagicMock()},
        "verifyfin.data_pipeline.edgar_client": {"EdgarClient": MagicMock()},
        "verifyfin.data_pipeline.edgar_xbrl_client": {"EdgarXbrlClient": MagicMock()},
        "verifyfin.data_pipeline.kb_indexer": {"index_macro_data": MagicMock(), "index_news_headlines": MagicMock()},
        "verifyfin.data_pipeline.vector_store": {"VectorStore": MagicMock()},
        "verifyfin.graph.pipeline": {"VerifyFinPipeline": MagicMock()},
        "verifyfin.llm.groq_client": {"GroqClient": MagicMock()},
    }
    saved = {}
    for name, attrs in stubs.items():
        saved[name] = sys.modules.get(name)
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
    spec = importlib.util.spec_from_file_location("run_eval_h", SCRIPTS / "run_evaluation.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    finally:
        # The module keeps its own references to the stubs; put the real modules back so no other test sees a stub.
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    mod.resolve_path = lambda rel: pathlib.Path(tmp) / rel
    mod._RESULTS_TIMESTAMP = None
    return mod


class FM:
    def __init__(self, t, c):
        self.t, self.c = t, c

    def model_dump(self):
        return {"ticker": "AAPL", "time_period": self.t, "condition": self.c, "total_claims": 3}


SCEN = [{"ticker": "AAPL", "period_end": "2026-06-27", "accession_number": "a", "time_period_label": "AAPL-Q2"},
        {"ticker": "AAPL", "period_end": "2026-03-28", "accession_number": "b", "time_period_label": "AAPL-Q1"}]


def wire(mod, calls):
    mod.load_config = lambda: {"tickers": {"evaluation": ["AAPL"]}}
    mod.resolve_scenarios = lambda tickers, periods_per_ticker=3: SCEN
    mod.build_knowledge_base_for_scenario = lambda sc: calls.append(("build", sc["time_period_label"])) or object()
    mod.open_existing_knowledge_base = lambda sc: calls.append(("reuse", sc["time_period_label"])) or object()

    def mk(cond, with_claims=False):
        def runner(sc, store):
            calls.append((cond, sc["time_period_label"]))
            out = {"report": f"{cond} report", "metrics": FM(sc["time_period_label"], cond)}
            if with_claims:
                out["claims"] = [object(), object()]
            return out
        return runner

    mod.run_single_agent_condition = mk("single_agent")
    mod.run_unverified_condition = mk("unverified_multiagent")
    mod.run_verify_fin_condition = mk("verify_fin", True)
    mod.run_single_agent_rag_condition = mk("single_agent_rag", True)
    mod._claim_to_record = lambda sc, claim: {"claim_id": f"{sc['time_period_label']}-{id(claim) % 9973}", "ticker": sc["ticker"]}


def conditions(out):
    f = sorted(out.glob("evaluation_results_*.json"))[-1]
    rows = json.load(open(f))
    return sorted((r["time_period"], r["condition"]) for r in rows), f


class TestFreshRun:
    def test_default_run_has_three_conditions_and_no_rag(self, tmp_path):
        mod = _load(tmp_path)
        calls = []
        wire(mod, calls)
        mod.main()
        rows, _ = conditions(tmp_path / "outputs/results")
        assert {c for _, c in rows} == {"single_agent", "unverified_multiagent", "verify_fin"}
        assert not list((tmp_path / "outputs/results").glob("rag_claims_*"))

    def test_rag_flag_adds_a_fourth_condition_and_a_separate_claims_file(self, tmp_path):
        mod = _load(tmp_path)
        wire(mod, [])
        mod.main(with_rag_baseline=True)
        out = tmp_path / "outputs/results"
        rows, _ = conditions(out)
        assert {c for _, c in rows} == {"single_agent", "unverified_multiagent", "verify_fin", "single_agent_rag"} and len(rows) == 8
        assert len(list(out.glob("rag_claims_*.jsonl"))) == 1

    def test_rag_claims_file_can_never_be_mistaken_for_verify_fin_claims(self, tmp_path):
        """The sampler / analysis / audit scripts all pick sorted(glob('claims_*.jsonl'))[-1]."""
        mod = _load(tmp_path)
        wire(mod, [])
        mod.main(with_rag_baseline=True)
        out = tmp_path / "outputs/results"
        matches = sorted(out.glob("claims_*.jsonl"))
        assert len(matches) == 1 and "rag" not in matches[0].name
        n_vf = sum(1 for _ in open(matches[0]))
        assert n_vf == 2 * len(SCEN)                       # only VERIFY-FIN's claims (2 per scenario in the stub)


class TestAddingRagToAnExistingRun:
    def _existing(self, mod, tmp_path):
        calls = []
        wire(mod, calls)
        mod.main()
        mod._RESULTS_TIMESTAMP = None
        return calls

    def test_fill_missing_with_rag_runs_only_the_new_condition_and_reuses_the_knowledge_bases(self, tmp_path):
        mod = _load(tmp_path)
        self._existing(mod, tmp_path)
        out = tmp_path / "outputs/results"
        before, f = conditions(out)
        main_claims_before = (next(out.glob("claims_*.jsonl"))).read_text()
        calls = []
        wire(mod, calls)
        mod.main(fill_missing=True, with_rag_baseline=True)
        assert calls == [("reuse", "AAPL-Q2"), ("single_agent_rag", "AAPL-Q2"), ("reuse", "AAPL-Q1"), ("single_agent_rag", "AAPL-Q1")]
        assert not [c for c in calls if c[0] == "build"]
        after, f2 = conditions(out)
        assert f2 == f                                                     # same timestamped file, updated in place
        assert len(after) == len(before) + 2 and sum(1 for _, c in after if c == "single_agent_rag") == 2
        assert (next(out.glob("claims_*.jsonl"))).read_text() == main_claims_before   # VERIFY-FIN claims untouched

    def test_rerunning_is_idempotent_so_an_interrupted_run_can_be_resumed(self, tmp_path):
        mod = _load(tmp_path)
        self._existing(mod, tmp_path)
        wire(mod, [])
        mod.main(fill_missing=True, with_rag_baseline=True)
        mod._RESULTS_TIMESTAMP = None
        calls = []
        wire(mod, calls)
        mod.main(fill_missing=True, with_rag_baseline=True)
        assert calls == []                                                 # nothing left to do

    def test_partially_finished_rag_run_resumes_where_it_stopped(self, tmp_path):
        mod = _load(tmp_path)
        self._existing(mod, tmp_path)
        wire(mod, [])
        # first RAG pass crashes on the second scenario
        original = mod.run_single_agent_rag_condition

        def flaky(sc, store):
            if sc["time_period_label"] == "AAPL-Q1":
                raise RuntimeError("rate limit")
            return original(sc, store)

        mod.run_single_agent_rag_condition = flaky
        mod.main(fill_missing=True, with_rag_baseline=True)
        mod._RESULTS_TIMESTAMP = None
        calls = []
        wire(mod, calls)
        mod.main(fill_missing=True, with_rag_baseline=True)
        assert calls == [("reuse", "AAPL-Q1"), ("single_agent_rag", "AAPL-Q1")]

    def test_existing_rag_claims_are_preserved_when_resuming(self, tmp_path):
        mod = _load(tmp_path)
        self._existing(mod, tmp_path)
        wire(mod, [])
        mod.main(fill_missing=True, with_rag_baseline=True)
        out = tmp_path / "outputs/results"
        n1 = sum(1 for _ in open(next(out.glob("rag_claims_*.jsonl"))))
        mod._RESULTS_TIMESTAMP = None
        wire(mod, [])
        mod.main(fill_missing=True, with_rag_baseline=True)
        assert sum(1 for _ in open(next(out.glob("rag_claims_*.jsonl")))) == n1

    def test_a_mixed_gap_still_rebuilds_the_knowledge_base(self, tmp_path):
        """If another condition is also missing, the original (rebuild) behaviour applies."""
        mod = _load(tmp_path)
        self._existing(mod, tmp_path)
        out = tmp_path / "outputs/results"
        f = sorted(out.glob("evaluation_results_*.json"))[-1]
        rows = [r for r in json.load(open(f)) if not (r["time_period"] == "AAPL-Q1" and r["condition"] == "verify_fin")]
        json.dump(rows, open(f, "w"))
        calls = []
        wire(mod, calls)
        mod._RESULTS_TIMESTAMP = None
        mod.main(fill_missing=True, with_rag_baseline=True)
        assert ("build", "AAPL-Q1") in calls and ("verify_fin", "AAPL-Q1") in calls and ("single_agent_rag", "AAPL-Q1") in calls
        assert ("reuse", "AAPL-Q2") in calls
