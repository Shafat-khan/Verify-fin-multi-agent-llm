"""
Tests for scripts/audit_metrics.py.

The synthetic dataset reproduces the project's real human-evaluation
structure exactly: 171 claims, confusion cells TP=154 FP=3 FN=5 TN=9, the
descriptive table (GROUNDED/UPHELD/REJECTED x label), the category split
(101 numerical / 56 macro / 14 sentiment) and the cluster structure
(macro 20 distinct of 56, sentiment 5 distinct of 14).
"""

import csv
import importlib.util
import json
import pathlib

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"


def _load_audit():
    spec = importlib.util.spec_from_file_location("audit_metrics", SCRIPTS / "audit_metrics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_dataset(tmp_path):
    # (category, outcome bucket, human label) -> count; totals match the real run
    cells = (
        [("numerical", "GROUNDED", "SUPPORTED")] * 88 + [("numerical", "UPHELD", "SUPPORTED")] * 13
        + [("macro", "GROUNDED", "SUPPORTED")] * 40 + [("macro", "GROUNDED", "CONTRADICTED")] * 2 + [("macro", "GROUNDED", "UNSUPPORTED")] * 1
        + [("macro", "UPHELD", "SUPPORTED")] * 10 + [("macro", "REJECTED", "SUPPORTED")] * 2 + [("macro", "REJECTED", "UNSUPPORTED")] * 1
        + [("sentiment", "UPHELD", "SUPPORTED")] * 3 + [("sentiment", "REJECTED", "SUPPORTED")] * 3 + [("sentiment", "REJECTED", "UNSUPPORTED")] * 8
    )
    assert len(cells) == 171
    sizes = [13] * 6 + [12] * 7 + [9]
    tickers = ["AAPL"] * 3 + ["MSFT"] * 3 + ["JPM"] * 3 + ["JNJ"] * 3 + ["XOM"] * 2
    scen = [(tickers[i], f"{tickers[i]}-S{i}") for i in range(14)]
    # interleave so every scenario gets a mix
    order = sorted(range(171), key=lambda i: (i * 7919) % 171)
    claims, k = [], 0
    macro_texts = [f"Macro fact {j}" for j in range(20)]
    sent_texts = ["AAPL sent", "MSFT sent", "JNJ sent", "JPM sent", "XOM sent"]
    m_i = s_i = n_i = 0
    sidx = 0
    for si, size in enumerate(sizes):
        for _ in range(size):
            cat, bucket, label = cells[order[k]]
            k += 1
            if cat == "numerical":
                text = f"Numerical claim {n_i}"; n_i += 1
            elif cat == "macro":
                text = macro_texts[m_i % 20]; m_i += 1
            else:
                text = sent_texts[s_i % 5]; s_i += 1
            t, tp = scen[si]
            usable = bucket in ("GROUNDED", "UPHELD")
            claims.append(dict(claim_id=f"c{k}", ticker=t, time_period=tp, claim_category=cat, claim_text=text,
                               retrieved_evidence="ev", grounding_score=0.8,
                               verdict="grounded" if bucket == "GROUNDED" else "challenged",
                               debate_resolution={"UPHELD": "upheld", "REJECTED": "rejected"}.get(bucket),
                               is_usable=usable, _label=label, _bucket=bucket))
    results = []
    for t, tp in scen:
        cs = [c for c in claims if c["time_period"] == tp]
        base = dict(ticker=t, time_period=tp, runtime_seconds=5.0, llm_total_tokens=3000, llm_api_call_count=3, llm_call_count=4,
                    total_claims=0, usable_claims_count=0, posthoc_total_claims=None, posthoc_usable_count=None)
        results.append({**base, "condition": "verify_fin", "total_claims": len(cs), "usable_claims_count": sum(c["is_usable"] for c in cs)})
        results.append({**base, "condition": "unverified_multiagent", "posthoc_total_claims": len(cs), "posthoc_usable_count": len(cs) - 1})
        results.append({**base, "condition": "single_agent", "posthoc_total_claims": 15, "posthoc_usable_count": 0, "llm_total_tokens": 700})
    rdir = pathlib.Path(tmp_path)
    json.dump(results, open(rdir / "res.json", "w"))
    with open(rdir / "claims.jsonl", "w") as f:
        for c in claims:
            f.write(json.dumps({k_: v for k_, v in c.items() if not k_.startswith("_")}) + "\n")
    with open(rdir / "ws.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["claim_id", "ticker", "time_period", "claim_category", "claim_text", "retrieved_evidence", "human_label", "notes"])
        w.writeheader()
        for c in claims:
            w.writerow({"claim_id": c["claim_id"], "ticker": c["ticker"], "time_period": c["time_period"], "claim_category": c["claim_category"],
                        "claim_text": c["claim_text"], "retrieved_evidence": "ev", "human_label": c["_label"], "notes": ""})
    key = {c["claim_id"]: dict(verdict=c["verdict"], debate_resolution=c["debate_resolution"], grounding_score=0.8,
                               is_usable=c["is_usable"], outcome_bucket=c["_bucket"]) for c in claims}
    json.dump(key, open(rdir / "key.json", "w"))
    return {k_: str(rdir / v) for k_, v in dict(results="res.json", claims="claims.jsonl", worksheet="ws.csv", answer_key="key.json").items()}


@pytest.fixture
def dataset(tmp_path):
    return build_dataset(tmp_path)


class TestAuditOnConsistentData:
    def test_no_failures_and_known_values(self, dataset):
        au = _load_audit()
        d = au.load_inputs(dataset["results"], dataset["claims"], dataset["worksheet"], dataset["answer_key"])
        a = au.run_audit(d, exclude_cost=[], reported=False)
        fails = [r for r in a.rows if r[0] == "FAIL"]
        assert not fails, fails
        text = " ".join(r[1] + " " + r[2] for r in a.rows)
        assert "TP=154 FP=3 FN=5 TN=9" in text
        assert "126 distinct claims from 171 rows" in text
        assert "unsupported share all 7.0175%" in text
        assert "removed 9/12 unsupported, lost 5/159 supported" in text

    def test_independent_numbers_match_hand_calculation(self, dataset):
        au = _load_audit()
        d = au.load_inputs(dataset["results"], dataset["claims"], dataset["worksheet"], dataset["answer_key"])
        a = au.Audit()
        h = au.human_recompute(d, a)
        assert h["accuracy"] == pytest.approx(163 / 171)
        assert h["kappa"] == pytest.approx(0.667, abs=1e-3)
        assert h["mcnemar_p"] == pytest.approx(0.424, abs=1e-3)
        assert h["ablation_accuracy_no_debate"] == pytest.approx(137 / 171)
        assert h["ablation_kappa_no_debate"] == pytest.approx(0.267, abs=1e-3)
        assert h["unsupported_kept"] == pytest.approx(3 / 157)

    def test_bootstrap_intervals_are_stable_and_ordered(self, dataset):
        au = _load_audit()
        d = au.load_inputs(dataset["results"], dataset["claims"], dataset["worksheet"], dataset["answer_key"])
        h = au.human_recompute(d, au.Audit())
        assert h["cluster_acc_lo"] < 163 / 171 < h["cluster_acc_hi"]
        assert h["cluster_reduction_lo"] > -0.05 and h["cluster_reduction_hi"] < 0.2


class TestAuditDetectsProblems:
    def test_detects_results_vs_claims_mismatch(self, dataset, tmp_path):
        au = _load_audit()
        res = json.load(open(dataset["results"]))
        for r in res:
            if r["condition"] == "verify_fin":
                r["usable_claims_count"] -= 1
                break
        json.dump(res, open(dataset["results"], "w"))
        d = au.load_inputs(dataset["results"], dataset["claims"], dataset["worksheet"], dataset["answer_key"])
        a = au.Audit(); au.check_integrity(d, a)
        assert any(s == "FAIL" and "results vs claims file agree" in n for s, n, _ in a.rows)

    def test_detects_answer_key_id_mismatch(self, dataset):
        au = _load_audit()
        key = json.load(open(dataset["answer_key"]))
        key["ghost"] = key.pop(next(iter(key)))
        json.dump(key, open(dataset["answer_key"], "w"))
        d = au.load_inputs(dataset["results"], dataset["claims"], dataset["worksheet"], dataset["answer_key"])
        a = au.Audit(); au.check_integrity(d, a)
        assert any(s == "FAIL" and "worksheet ids == answer-key ids" in n for s, n, _ in a.rows)

    def test_detects_invalid_label(self, dataset):
        au = _load_audit()
        rows = list(csv.DictReader(open(dataset["worksheet"], newline="")))
        rows[0]["human_label"] = "MAYBE"
        with open(dataset["worksheet"], "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        d = au.load_inputs(dataset["results"], dataset["claims"], dataset["worksheet"], dataset["answer_key"])
        a = au.Audit(); au.check_integrity(d, a)
        assert any(s == "FAIL" and "valid label" in n for s, n, _ in a.rows)

    def test_detects_inconsistent_usability_flag(self, dataset):
        au = _load_audit()
        claims = [json.loads(l) for l in open(dataset["claims"])]
        claims[0]["is_usable"] = not claims[0]["is_usable"]
        with open(dataset["claims"], "w") as f:
            for c in claims: f.write(json.dumps(c) + "\n")
        d = au.load_inputs(dataset["results"], dataset["claims"], dataset["worksheet"], dataset["answer_key"])
        a = au.Audit(); au.check_integrity(d, a)
        assert any(s == "FAIL" for s, _, _ in a.rows)


class TestKnownMetricIssuesAreFlagged:
    def test_exact_p_and_both_effect_sizes_on_hand_computable_data(self):
        """
        100 claims per scenario so rates are exact. Bad-claim counts, baseline vs VERIFY-FIN:
        scenario 0: 30 vs 10 (diff +0.20), scenario 1: 20 vs 5 (+0.15), scenario 2: 5 vs 15 (-0.10), 11 scenarios 10 vs 10.
        Ranks of |diff|: 0.10 -> 1 (negative), 0.15 -> 2, 0.20 -> 3  =>  T+ = 5, T- = 1.
        Exact one-sided p = P(T >= 5) = 2/8 = 0.25; rank-biserial = (5-1)/6 = 0.667; net sign advantage = (2-1)/3 = 0.333.
        """
        au = _load_audit()
        base_bad = [30, 20, 5] + [10] * 11
        vf_bad = [10, 5, 15] + [10] * 11
        res = []
        for i, (b, v) in enumerate(zip(base_bad, vf_bad)):
            res.append(dict(ticker="T", time_period=f"S{i}", condition="verify_fin", total_claims=100, usable_claims_count=100 - v,
                            runtime_seconds=1, llm_total_tokens=1, llm_api_call_count=1))
            res.append(dict(ticker="T", time_period=f"S{i}", condition="unverified_multiagent", posthoc_total_claims=100,
                            posthoc_usable_count=100 - b, runtime_seconds=1, llm_total_tokens=1, llm_api_call_count=1))
        a = au.Audit()
        sc = au.scenario_stats({"results": res}, a)
        paired = au.paired_checks({"results": res}, sc, a)
        u = paired["unverified_multiagent"]
        assert u["n_nonzero"] == 3
        assert u["exact_p"] == pytest.approx(0.25)
        assert u["rank_biserial"] == pytest.approx(2 / 3)
        assert u["sign_net"] == pytest.approx(1 / 3)
        assert not [r for r in a.rows if r[0] == "FAIL"]

    def test_reported_effect_size_and_p_are_flagged_as_misleading(self):
        """The previously quoted 0.2071 / 0.333 must be flagged when they disagree with the exact / rank-biserial values."""
        au = _load_audit()
        a = au.Audit()
        paired = {"unverified_multiagent": {"scipy_p": 0.2071, "exact_p": 0.375, "rank_biserial": 0.5, "sign_net": 1 / 3, "n_nonzero": 3}}
        au.compare_reported({}, paired, None, a)
        warns = [n for s, n, _ in a.rows if s == "WARN"]
        assert any("Wilcoxon p" in n for n in warns) and any("effect size" in n for n in warns)
