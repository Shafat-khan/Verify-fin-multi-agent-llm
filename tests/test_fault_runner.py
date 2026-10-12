"""
End-to-end tests for verifyfin.evaluation.fault_runner with the REAL VerificationAgent / ClaimObject over a fake
knowledge base. Similarity is deterministic: sim = 0.45 + 0.4 * Jaccard(word tokens), so every verdict below can
be derived by hand (e.g. a clean revenue claim: Jaccard 0.40 -> sim 0.61; numeric match boosts +0.25 -> 0.86 GROUNDED;
without the boost 0.61 is CHALLENGED).
"""

import re

import pytest

from verifyfin.evaluation import fault_injection as fi
from verifyfin.evaluation import fault_runner as fr

TP = "AAPL-2026-Q2"
CHUNKS = [
    ("Total net sales 109,417 for the three months ended June 27, 2026", "edgar_10q"),
    ("Net income 29,789 for the three months ended June 27, 2026", "edgar_10q"),
    ("FEDFUNDS observation: 3.63 on 2026-07-01", "fred"),
    ("UNRATE observation: 4.1 on 2026-07-01", "fred"),
    ("Headline: Apple shares fall on supply worries", "news"),
]


def _tokens(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


class FakeStore:
    def __init__(self):
        self.chunks = [{"text": t, "metadata": {"ticker": "AAPL", "source_type": st}} for t, st in CHUNKS]
        self.calls = 0

    def query(self, query_text, ticker=None, top_k=5, where_document_contains=None):
        self.calls += 1
        q = _tokens(query_text)
        out = []
        for c in self.chunks:
            if ticker and c["metadata"]["ticker"] != ticker:
                continue
            if where_document_contains and where_document_contains not in c["text"]:
                continue
            t = _tokens(c["text"])
            jac = len(q & t) / len(q | t) if q | t else 0.0
            out.append({"text": c["text"], "metadata": c["metadata"], "similarity_score": round(0.45 + 0.4 * jac, 4)})
        return sorted(out, key=lambda r: -r["similarity_score"])[:top_k]

    def count(self):
        return len(self.chunks)


class FakeDebate:
    def __init__(self):
        self.calls = 0

    def debate_claim(self, claim):
        self.calls += 1
        claim.debate_resolution = "rejected" if self.calls % 2 else "upheld"
        return claim


N1 = "Apple reported total revenue of $109,417,000,000 for the period ending June 27, 2026."
M1 = "On 2026-07-01, the Federal Funds Rate was 3.63."
M_BAD = "On 2026-07-01, the U.S. unemployment rate was 9.9."


def worksheet():
    def row(i, cat, text, label):
        return dict(claim_id=f"c{i}", ticker="AAPL", time_period=TP, claim_category=cat, claim_text=text, human_label=label)
    return [row(1, "numerical", N1, "SUPPORTED"), row(2, "macro", M1, "SUPPORTED"), row(3, "macro", M_BAD, "UNSUPPORTED"),
            row(4, "sentiment", "Headlines convey negative sentiment.", "SUPPORTED"),
            row(5, "numerical", "Revenue grew strongly.", "SUPPORTED"),
            row(6, "macro", "On 2026-07-01, the CPI was 332.813.", "")]


def bases():
    b, skipped = fr.build_base_claims(worksheet(), {}, {TP: "2026-06-27"})
    return b, skipped


def run(configs=fr.DEFAULT_CONFIGS, **kw):
    b, _ = bases()
    return fr.run_experiment(b, lambda tp: FakeStore(), configs=configs, log=lambda m: None, **kw)


def detection(res, config, **sel):
    rows = [r for r in res["rows"] if r["config"] == config and r["kind"] == "corrupted"
            and all(r[k] == v for k, v in sel.items())]
    assert rows, (config, sel)
    return sum(1 for r in rows if r["verdict"] != "grounded") / len(rows)


class TestBaseClaims:
    def test_selection_and_skips(self):
        b, skipped = bases()
        assert [x["claim_id"] for x in b] == ["c1", "c2", "c3"]
        assert skipped == {"category": 1, "no_excerpt": 1, "no_label": 1}

    def test_human_labels_and_excerpts(self):
        b, _ = bases()
        by = {x["claim_id"]: x for x in b}
        assert by["c1"]["human_supported"] and not by["c3"]["human_supported"]
        assert by["c1"]["excerpt"] == "Reconstructed: 109417000000 USD (period N/A to 2026-06-27)"
        assert by["c2"]["excerpt"] == "Reconstructed: 3.63 (as of 2026-07-01)"


class TestPrimaryConfiguration:
    def test_clean_claims_are_grounded_and_unsupported_original_is_not(self):
        res = run()
        ctl = {r["base_id"]: r for r in res["rows"] if r["config"] == "full|all" and r["kind"] == "control"}
        assert ctl["c1"]["verdict"] == "grounded" and ctl["c2"]["verdict"] == "grounded"
        assert ctl["c3"]["verdict"] != "grounded"                       # 9.9 appears nowhere in the knowledge base

    def test_only_supported_claims_are_corrupted(self):
        res = run()
        assert res["fi_ids"] == {"c1", "c2"} and {v["base_id"] for v in res["variants"]} == {"c1", "c2"}

    def test_source_faults_are_caught_phrasing_faults_are_missed(self):
        """The central, honest finding about the verifier's design: it takes the expected number from the EXCERPT."""
        res = run()
        for fam in ("magnitude", "typo"):
            assert detection(res, "full|all", family=fam, mode="source", category="numerical") == 1.0
            assert detection(res, "full|all", family=fam, mode="phrasing", category="numerical") == 0.0

    def test_label_swap_is_a_blind_spot(self):
        res = run()
        assert detection(res, "full|all", family="attribution") == 0.0

    def test_wrong_date_macro_claim_slips_through_the_default_verifier(self):
        res = run()
        assert detection(res, "full|all", operator="date_shift_month", mode="source") == 0.0

    def test_every_row_has_a_score_in_range(self):
        res = run()
        assert all(0.0 <= r["score"] <= 1.0 for r in res["rows"])


class TestAblation:
    def test_strict_date_catches_what_the_default_misses(self):
        res = run()
        assert detection(res, "strict_date|all", operator="date_shift_month", mode="source") == 1.0
        assert detection(res, "strict_date|all", operator="date_shift_year", mode="source") == 1.0

    def test_strict_date_does_not_hurt_clean_claims(self):
        res = run()
        ctl = [r for r in res["rows"] if r["config"] == "strict_date|all" and r["kind"] == "control" and r["in_fi"]]
        assert all(r["verdict"] == "grounded" for r in ctl)

    def test_strict_date_cannot_fix_phrasing_or_label_faults(self):
        res = run()
        assert detection(res, "strict_date|all", operator="date_shift_month", mode="phrasing") == 0.0
        assert detection(res, "strict_date|all", operator="series_swap") == 0.0

    def test_semantic_only_loses_the_numeric_boost_and_flags_clean_claims(self):
        res = run()
        ctl = [r for r in res["rows"] if r["config"] == "semantic_only|all" and r["kind"] == "control" and r["in_fi"]]
        assert all(r["verdict"] == "challenged" for r in ctl)           # 0.61 / 0.63 without the +0.25 boost

    def test_no_boost_removes_exactly_the_boost_but_keeps_the_numerically_matched_evidence(self):
        """
        no_boost still PICKS the chunk containing the number (unlike semantic_only, which always takes the top-ranked chunk),
        so on clean claims its score is the full score minus the 0.25 boost, and it can never exceed the full score.
        """
        res = run()
        full = {(r["base_id"], r["kind"], r["operator"], r["mode"]): r["score"] for r in res["rows"] if r["config"] == "full|all"}
        nb = {(r["base_id"], r["kind"], r["operator"], r["mode"]): r["score"] for r in res["rows"] if r["config"] == "no_boost|all"}
        assert all(nb[k] <= full[k] + 1e-9 for k in full)
        for cid in ("c1", "c2"):
            k = (cid, "control", "-", "-")
            assert full[k] - nb[k] == pytest.approx(0.25, abs=1e-3)

    def test_semantic_only_equals_no_boost_when_the_matched_chunk_is_also_top_ranked(self):
        res = run()
        sem = {r["base_id"]: r["score"] for r in res["rows"] if r["config"] == "semantic_only|all" and r["kind"] == "control"}
        nb = {r["base_id"]: r["score"] for r in res["rows"] if r["config"] == "no_boost|all" and r["kind"] == "control"}
        assert sem["c1"] == nb["c1"] and sem["c2"] == nb["c2"]

    def test_knowledge_base_ablation_without_fred_breaks_macro_controls(self):
        res = run()
        macro_ctl = [r for r in res["rows"] if r["config"] == "full|filing_only" and r["kind"] == "control" and r["base_id"] == "c2"]
        assert macro_ctl[0]["verdict"] != "grounded"                    # no FRED chunks -> nothing to ground a macro claim on
        macro_ok = [r for r in res["rows"] if r["config"] == "full|filing_fred" and r["kind"] == "control" and r["base_id"] == "c2"]
        assert macro_ok[0]["verdict"] == "grounded"

    def test_human_agreement_is_computed_for_every_configuration(self):
        res = run()
        for cfg in res["configs"]:
            ha = fr.human_agreement(res["rows"], cfg)
            assert ha["n"] == 3 and ha["tp"] + ha["fp"] + ha["fn"] + ha["tn"] == 3
        assert fr.human_agreement(res["rows"], "full|all")["accuracy"] == pytest.approx(1.0)

    def test_variant_caching_shares_retrievals_across_configurations(self):
        calls = []

        class Counting(FakeStore):
            def query(self, *a, **k):
                calls.append(1)
                return super().query(*a, **k)

        b, _ = bases()
        fr.run_experiment(b, lambda tp: Counting(), configs=[("full", "all")], log=lambda m: None)
        single = len(calls)
        calls.clear()
        fr.run_experiment(b, lambda tp: Counting(), configs=fr.DEFAULT_CONFIGS, log=lambda m: None)
        assert len(calls) < single * len(fr.DEFAULT_CONFIGS) * 0.6        # 8 configurations cost far less than 8x one


class TestValidatorAndOptions:
    def test_validator_drops_are_counted_and_variants_removed(self):
        res = run(validator_factory=lambda tp: (lambda base, op, text, meta: False if op == "series_swap" else True))
        assert res["drops"]["not_false"] == 1
        assert not [v for v in res["variants"] if v["operator"] == "series_swap"]

    def test_operator_filter_and_max_bases(self):
        res = run(operators=["scale_5pct"])
        assert {v["operator"] for v in res["variants"]} == {"scale_5pct"}
        res2 = run(max_bases=1)
        assert len(res2["fi_ids"]) == 1


class TestFidelity:
    def test_perfect_when_original_scores_match(self):
        b, _ = bases()
        res0 = fr.run_experiment(b, lambda tp: FakeStore(), configs=[("full", "all")], log=lambda m: None)
        ctl = {r["base_id"]: r for r in res0["rows"] if r["kind"] == "control"}
        for x in b:
            x["orig_score"], x["orig_verdict"] = ctl[x["claim_id"]]["score"], ctl[x["claim_id"]]["verdict"]
        res = fr.run_experiment(b, lambda tp: FakeStore(), configs=[("full", "all")], log=lambda m: None)
        assert res["fidelity"]["rate"] == 1.0 and res["fidelity"]["n"] == 3

    def test_mismatches_are_reported(self):
        b, _ = bases()
        for x in b:
            x["orig_score"], x["orig_verdict"] = 0.10, "blocked"
        res = fr.run_experiment(b, lambda tp: FakeStore(), configs=[("full", "all")], log=lambda m: None)
        assert res["fidelity"]["rate"] == 0.0 and res["fidelity"]["mismatches"]
        assert "WARNING: below 90%" in fr.format_report(res, n_boot=100)


class TestDebateSample:
    def test_sample_respects_budget_and_totals(self):
        dbg = FakeDebate()
        res = run(debate_agent=dbg, debate_budget=3)
        d = res["debate"]
        assert d["n_sampled"] == min(3, d["n_challenged_total"]) == dbg.calls
        assert d["rejected"] + d["upheld"] == d["n_sampled"]

    def test_no_debate_without_budget(self):
        assert "debate" not in run(debate_agent=FakeDebate(), debate_budget=0)


class TestReport:
    def test_report_contains_every_section_and_consistent_counts(self):
        res = run(debate_agent=FakeDebate(), debate_budget=4)
        text = fr.format_report(res, n_boot=200)
        for needle in ["FAULT-INJECTION EXPERIMENT", "RECONSTRUCTION FIDELITY", "PRIMARY RESULT", "By fault family and injection mode",
                       "FALSE-FLAG RATE", "SCORE SEPARATION", "AUROC", "THRESHOLD SWEEP", "DEBATE-STAGE SAMPLE",
                       "ABLATION: every verifier", "Detection by fault family", "READING THE ABLATION", "strict_date|all", "semantic_only|all"]:
            assert needle in text, needle
        n_cor = len([r for r in res["rows"] if r["config"] == "full|all" and r["kind"] == "corrupted"])
        assert f"ALL corrupted" in text and str(n_cor) in text
        assert n_cor == len(res["variants"])

    def test_csv_roundtrip(self, tmp_path):
        res = run()
        p = tmp_path / "rows.csv"
        fr.save_rows_csv(res["rows"], p)
        import csv
        back = list(csv.DictReader(open(p, newline="")))
        assert len(back) == len(res["rows"]) and set(back[0]) >= {"config", "kind", "score", "verdict", "operator"}


class TestTextConsistencyVariant:
    """v2 closes the documented blind spot: wrong SENTENCE, true excerpt ('phrasing' mode)."""

    CONFIGS = [("full", "all"), ("text_check", "all"), ("text_check_block", "all")]

    def test_phrasing_faults_are_now_flagged_but_clean_claims_are_not(self):
        res = run(configs=self.CONFIGS)
        for op in ("scale_25pct", "scale_10x", "digit_noise_1"):
            assert detection(res, "full|all", category="numerical", mode="phrasing", operator=op) == 0.0
            assert detection(res, "text_check|all", category="numerical", mode="phrasing", operator=op) == 1.0
        ctl = lambda cfg: [r["verdict"] for r in res["rows"] if r["config"] == cfg and r["kind"] == "control" and r["in_fi"]]
        assert ctl("text_check|all") == ctl("full|all")                      # no new false flags on the clean originals

    def test_source_mode_detection_is_not_made_worse(self):
        res = run(configs=self.CONFIGS)
        for op in ("scale_25pct", "digit_noise_1"):
            assert (detection(res, "text_check|all", category="numerical", mode="source", operator=op)
                    >= detection(res, "full|all", category="numerical", mode="source", operator=op))

    def test_block_variant_routes_mismatches_to_blocked(self):
        res = run(configs=self.CONFIGS)
        v = {r["verdict"] for r in res["rows"] if r["config"] == "text_check_block|all" and r["kind"] == "corrupted"
             and r["mode"] == "phrasing" and r["category"] == "numerical"}
        assert "blocked" in v

    def test_debate_on_the_v2_configuration_also_debates_challenged_originals(self):
        res = run(configs=self.CONFIGS, debate_agent=FakeDebate(), debate_budget=4, debate_config="text_check|all")
        d = res["debate"]
        assert d["config"] == "text_check|all" and d["n_sampled"] <= 4 and "controls" in d
        assert "DEBATE-STAGE SAMPLE (configuration 'text_check|all')" in fr.format_report(res, n_boot=50)

    def test_primary_debate_does_not_redebate_controls(self):
        res = run(configs=self.CONFIGS, debate_agent=FakeDebate(), debate_budget=4)
        assert "controls" not in res["debate"] and res["debate"]["config"] == "full|all"


class TestV3EvidenceAlignment:
    """v3 closes two blind spots the fault injection exposed: right-value-wrong-month and right-value-wrong-label."""

    CONFIGS = [("full", "all"), ("text_check", "all"), ("v3_date", "all"), ("v3_label", "all"), ("v3_aligned", "all"), ("v3_exact", "all")]

    def test_every_v3_config_is_a_known_variant(self):
        for v, k in fr.V3_CONFIGS:
            assert v in fr.VERIFIER_VARIANTS and k == "all"

    def test_wrong_date_macro_claims_are_caught_by_exact_record_lookup(self):
        res = run(configs=self.CONFIGS)
        checked = 0
        for op in ("date_shift_year", "date_shift_month"):
            sel = dict(category="macro", mode="source", operator=op)
            try:
                before = detection(res, "text_check|all", **sel)
                after = detection(res, "v3_date|all", **sel)
            except AssertionError:
                continue                     # operator not applicable to this tiny fake knowledge base
            checked += 1
            assert after == 1.0 and after >= before
        assert checked >= 1

    def test_series_swap_is_caught_by_the_label_check(self):
        res = run(configs=self.CONFIGS)
        assert detection(res, "v3_label|all", category="macro", operator="series_swap") == 1.0
        assert detection(res, "v3_label|all", category="macro", operator="series_swap") >= detection(res, "full|all", category="macro", operator="series_swap")

    def test_numeric_label_swap_is_caught_by_the_label_check(self):
        res = run(configs=self.CONFIGS)
        assert detection(res, "full|all", category="numerical", operator="label_swap") == 0.0          # the documented blind spot
        assert detection(res, "v3_label|all", category="numerical", operator="label_swap") == 1.0

    def test_no_new_false_flags_on_the_clean_claims(self):
        res = run(configs=self.CONFIGS)
        ctl = lambda cfg: {r["base_id"]: r["verdict"] for r in res["rows"] if r["config"] == cfg and r["kind"] == "control" and r["in_fi"]}
        base = ctl("text_check|all")
        for cfg in ("v3_date|all", "v3_label|all", "v3_aligned|all", "v3_exact|all"):
            for k, verdict in ctl(cfg).items():
                if base[k] == "grounded":
                    assert verdict == "grounded", (cfg, k)

    def test_exact_record_never_lowers_a_verdict(self):
        res = run(configs=self.CONFIGS)
        rank = {"blocked": 0, "challenged": 1, "grounded": 2}
        by = lambda cfg: {(r["kind"], r["base_id"], r["operator"], r["mode"]): rank[r["verdict"]] for r in res["rows"] if r["config"] == cfg}
        a, x = by("v3_aligned|all"), by("v3_exact|all")
        assert all(x[k] >= a[k] for k in a)

    def test_report_lists_v3_configurations(self):
        res = run(configs=self.CONFIGS)
        text = fr.format_report(res, n_boot=50)
        for name in ("v3_date|all", "v3_label|all", "v3_aligned|all", "v3_exact|all"):
            assert name in text
