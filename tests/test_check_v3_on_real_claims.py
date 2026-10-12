import importlib.util
import json
import pathlib

from verifyfin.agents.verification_agent import VerificationAgent
from verifyfin.schemas.claim import AgentRole, ClaimCategory, ClaimObject, SourceReference

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"


def _load():
    spec = importlib.util.spec_from_file_location("check_v3_on_real_claims", SCRIPTS / "check_v3_on_real_claims.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Store:
    chunks = [("FEDFUNDS observation: 3.64 on 2026-03-01", "fred", 0.90), ("FEDFUNDS observation: 3.64 on 2026-04-01", "fred", 0.40)] + \
             [(f"Headline {i}", "news", 0.85 - 0.01 * i) for i in range(7)]

    def query(self, q, ticker=None, top_k=5, where_document_contains=None):
        out = [{"text": t, "metadata": {"source_type": s}, "similarity_score": sim} for t, s, sim in self.chunks
               if not where_document_contains or where_document_contains in t]
        return sorted(out, key=lambda r: -r["similarity_score"])[:top_k]


def claim():
    return ClaimObject(claim_text="The Federal Funds Rate was 3.64 on 2026-04-01.", claim_category=ClaimCategory.MACRO,
                       produced_by=AgentRole.MACRO, ticker="AAPL", time_period="AAPL-2026-Q1", confidence_score=0.95,
                       source_reference=SourceReference(source_type="fred", document_id="FEDFUNDS", excerpt="FEDFUNDS: 3.64 (as of 2026-04-01)"))


def test_verify_all_runs_every_variant_on_fresh_copies(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    m = _load()
    c = claim()
    rows = m.verify_all([c], Store(), "AAPL-2026-Q1", lambda store, flags: VerificationAgent(vector_store=store, **flags))
    assert len(rows) == 1 and set(m.VARIANTS) <= set(rows[0])
    assert c.grounding_score is None                                  # the original claim object was not mutated (copies were verified)
    assert "2026-03-01" in rows[0]["v1"]["evidence"]                   # v1 cites the wrong month
    assert "2026-04-01" in rows[0]["v3"]["evidence"]                   # v3 repairs the citation


def test_summary_lists_repairs_and_flags_worse_cases():
    m = _load()
    mk = lambda v, s, e: {"verdict": v, "score": s, "evidence": e}
    rows = [{"scenario": "S1", "category": "macro", "claim_text": "better", "excerpt": "x", "v1": mk("grounded", .9, "a"), "v2": mk("grounded", .9, "a"),
             "v3": mk("grounded", .9, "b"), "v3x": mk("grounded", .9, "b")},
            {"scenario": "S2", "category": "numerical", "claim_text": "worse", "excerpt": "x", "v1": mk("grounded", .9, "a"),
             "v2": mk("grounded", .9, "a"), "v3": mk("challenged", .7, "a"), "v3x": mk("challenged", .7, "a")}]
    text = m.summarise(rows)
    assert "1 claims got WORSE" in text and "[WORSE]" in text and "[DIFFERENT PASSAGE]" in text and "claims verified: 2" in text


def test_main_end_to_end_with_fakes(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    m = _load()
    res = tmp_path / "r.json"
    res.write_text(json.dumps([{"ticker": "AAPL", "time_period": "AAPL-2026-Q1", "period_end": "2026-03-28"}]))
    out = tmp_path / "out.json"
    m.main(["--results", str(res), "--out", str(out)], analysts_factory=lambda: (lambda t, l, p: [claim()]),
           store_factory=lambda t, p: Store(), agent_factory=lambda store, flags: VerificationAgent(vector_store=store, **flags))
    assert json.load(open(out))[0]["scenario"] == "AAPL-2026-Q1"
    assert "v3 EVIDENCE-ALIGNMENT CHECK" in capsys.readouterr().out
