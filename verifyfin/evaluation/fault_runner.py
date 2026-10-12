"""
verifyfin.evaluation.fault_runner
====================================
Orchestration for the fault-injection experiment AND the ablation study
(faculty feedback points 1 and 6). Everything that touches the outside
world (the knowledge base, XBRL/FRED truth, the Debate LLM) is INJECTED, so
the whole flow is testable offline.

VERIFIER CONFIGURATIONS (ablation): each is a (verifier variant, knowledge-base view) pair
    full            all mechanisms on (what produced the reported results)
    semantic_only   no numeric/date logic at all = plain cosine, the proposal's original design
    no_date_pref    numeric match on the value only (ignores the claim's date)
    no_targeted     no second-stage substring search for the number
    no_boost        number found -> chosen as evidence, but no score boost
    strict_date     a FRED chunk counts as a numeric match only if it contains the claim's date
    text_check      v2: the number and date written in the SENTENCE must agree with the source excerpt
                    (mismatch -> CHALLENGED, so the Debate Agent decides); text_check_strict adds
                    strict_date; text_check_block sends a mismatch straight to BLOCKED
    v3_date         v2 + FRED claims must match the EXACT record "SERIES observation: value on DATE" (substring
                    search on the date), which also repairs a citation to the wrong month
    v3_label        v2 + the series / line item named in the sentence must match the matched passage
    v3_period       v2 + the filing-table column holding the number must carry the claimed period-end date
    v3_aligned      v2 + all three v3 checks;  v3_exact additionally routes a confirmed exact record to GROUNDED
    full|filing_only / full|filing_fred   knowledge-base ablation (no FRED/news, or no news)

WHAT "CAUGHT" MEANS
    Verification-stage detection = the corrupted claim is NOT routed to GROUNDED
    (it is BLOCKED outright, or CHALLENGED and handed to the Debate Agent). A
    GROUNDED corrupted claim is a certain miss: it reaches the report. A
    CHALLENGED one is only a MAYBE, because the Debate Agent may still uphold
    it. The optional Debate sample estimates the end-to-end detection rate.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from typing import Callable, Optional, Sequence

from verifyfin.evaluation import fault_injection as fi
from verifyfin.evaluation.stats_utils import cohens_kappa
from verifyfin.evaluation.store_wrappers import CachingStore, make_kb_view

VERIFIER_VARIANTS = {
    "full": {},
    "semantic_only": {"use_numeric_match": False},
    "no_date_pref": {"use_date_preference": False},
    "no_targeted": {"use_targeted_search": False},
    "no_boost": {"numeric_boost": 0.0},
    "strict_date": {"require_date_match": True},
    # v2: ALSO check that the number/date written in the sentence agree with the excerpt
    "text_check": {"check_text_consistency": True},
    "text_check_strict": {"check_text_consistency": True, "require_date_match": True},
    "text_check_block": {"check_text_consistency": True, "text_mismatch_action": "block"},
    # v3: evidence alignment (which passage was matched?), each piece on top of the v2 text check so gains are attributable
    "v3_date": {"check_text_consistency": True, "date_aligned_search": True},
    "v3_label": {"check_text_consistency": True, "label_check": True},
    "v3_period": {"check_text_consistency": True, "filing_period_check": True},
    "v3_aligned": {"check_text_consistency": True, "date_aligned_search": True, "label_check": True, "filing_period_check": True},
    "v3_exact": {"check_text_consistency": True, "date_aligned_search": True, "label_check": True, "filing_period_check": True,
                 "exact_record_grounds": True},
}
DEFAULT_CONFIGS = [
    ("full", "all"), ("semantic_only", "all"), ("no_date_pref", "all"), ("no_targeted", "all"),
    ("no_boost", "all"), ("strict_date", "all"), ("full", "filing_only"), ("full", "filing_fred"),
]
# The v1 ablation grid plus the v2 text-consistency variants (run with --configs, or --v2 on the CLI).
V2_CONFIGS = [("text_check", "all"), ("text_check_strict", "all"), ("text_check_block", "all")]
# The v3 evidence-alignment variants (run with --v3 on the CLI). text_check|all is the v2 reference they build on.
V3_CONFIGS = [("text_check", "all"), ("v3_date", "all"), ("v3_label", "all"), ("v3_period", "all"),
              ("v3_aligned", "all"), ("v3_exact", "all")]
PRIMARY = "full|all"


def cfg_name(variant: str, kb: str) -> str:
    return f"{variant}|{kb}"


# ------------------------------------------------------------------ base claims

def build_base_claims(worksheet: Sequence[dict], answer_key: dict, period_ends: dict,
                      categories: Sequence[str] = ("numerical", "macro")) -> tuple[list[dict], dict]:
    """Numerical and macro claims from the human-labelled worksheet, each with a reconstructed source excerpt."""
    bases, skipped = [], {"category": 0, "no_excerpt": 0, "no_label": 0}
    for r in worksheet:
        cat = r["claim_category"]
        if cat not in categories:
            skipped["category"] += 1
            continue
        label = (r.get("human_label") or "").strip().upper()
        if not label:
            skipped["no_label"] += 1
            continue
        pe = period_ends.get(r["time_period"])
        text = r["claim_text"].strip()
        excerpt = fi.reconstruct_excerpt(cat, text, pe)
        if excerpt is None:
            skipped["no_excerpt"] += 1
            continue
        ak = answer_key.get(r["claim_id"], {})
        bases.append(dict(claim_id=r["claim_id"], ticker=r["ticker"], time_period=r["time_period"], period_end=pe, category=cat,
                          text=text, excerpt=excerpt, human_label=label, human_supported=(label == "SUPPORTED"),
                          orig_score=ak.get("grounding_score"), orig_verdict=(ak.get("verdict") or "").lower() or None,
                          orig_usable=ak.get("is_usable"), orig_bucket=ak.get("outcome_bucket")))
    return bases, skipped


# ------------------------------------------------------------------ verification

def _claim_classes():
    from verifyfin.schemas.claim import AgentRole, ClaimCategory, ClaimObject, SourceReference
    return ClaimObject, ClaimCategory, AgentRole, SourceReference


def make_claim(item: dict):
    ClaimObject, ClaimCategory, AgentRole, SourceReference = _claim_classes()
    numerical = item["category"] == "numerical"
    return ClaimObject(
        claim_text=item["text"], claim_category=ClaimCategory(item["category"]),
        produced_by=AgentRole.FUNDAMENTAL if numerical else AgentRole.MACRO,
        source_reference=SourceReference(source_type="edgar_xbrl" if numerical else "fred", document_id="fault-injection",
                                         excerpt=item["excerpt"]),
        confidence_score=0.95, ticker=item["ticker"], time_period=item["time_period"])


def _verdict_str(claim) -> str:
    v = claim.verdict
    return v.value if hasattr(v, "value") else str(v)


def run_experiment(bases: list[dict], get_store: Callable[[str], object], *, configs: Sequence[tuple[str, str]] = DEFAULT_CONFIGS,
                   seed: int = 42, validator_factory: Optional[Callable[[str], Optional[fi.Validator]]] = None,
                   max_bases: Optional[int] = None, debate_agent=None, debate_budget: int = 0,
                   operators: Optional[Sequence[str]] = None, log: Callable[[str], None] = print,
                   debate_config: str = PRIMARY, debate_controls_budget: int = 40) -> dict:
    from verifyfin.agents.verification_agent import VerificationAgent

    rng = random.Random(seed)
    fi_bases = [b for b in bases if b["human_supported"]]
    if max_bases and len(fi_bases) > max_bases:
        fi_bases = rng.sample(fi_bases, max_bases)
    fi_ids = {b["claim_id"] for b in fi_bases}

    # ---- generate validated corruptions per scenario
    validators: dict[str, Optional[fi.Validator]] = {}
    variants, drops = [], defaultdict(int)
    for b in fi_bases:
        if b["time_period"] not in validators:
            validators[b["time_period"]] = validator_factory(b["time_period"]) if validator_factory else None
        vs, d = fi.generate_variants(b, seed=seed, validator=validators[b["time_period"]], operators=operators)
        variants.extend(vs)
        for k, n in d.items():
            drops[k] += n
    log(f"Base claims: {len(bases)} human-labelled numerical+macro ({len(fi_bases)} judged SUPPORTED are corrupted); "
        f"{len(variants)} corrupted variants generated; dropped {dict(drops)}")

    # ---- verify every config on controls (all bases) and corrupted variants
    stores = {}
    rows: list[dict] = []
    scenarios = sorted({b["time_period"] for b in bases} | {v["time_period"] for v in variants})
    for tp in scenarios:
        stores[tp] = CachingStore(get_store(tp))
    for variant, kb in configs:
        name = cfg_name(variant, kb)
        flags = VERIFIER_VARIANTS[variant]
        agents = {tp: VerificationAgent(vector_store=make_kb_view(stores[tp], kb), **flags) for tp in scenarios}
        for b in bases:
            c = make_claim(b)
            agents[b["time_period"]].verify_claim(c)
            rows.append(dict(config=name, kind="control", base_id=b["claim_id"], ticker=b["ticker"], time_period=b["time_period"],
                             category=b["category"], operator="-", family="-", mode="-", score=c.grounding_score,
                             verdict=_verdict_str(c), human_supported=b["human_supported"], in_fi=b["claim_id"] in fi_ids,
                             orig_usable=b["orig_usable"]))
        for v in variants:
            c = make_claim(v)
            agents[v["time_period"]].verify_claim(c)
            rows.append(dict(config=name, kind="corrupted", base_id=v["base_id"], ticker=v["ticker"], time_period=v["time_period"],
                             category=v["category"], operator=v["operator"], family=v["family"], mode=v["mode"], score=c.grounding_score,
                             verdict=_verdict_str(c), human_supported=True, in_fi=True, orig_usable=None))
        log(f"  verified config {name}: {len(bases)} controls + {len(variants)} corrupted")

    result = {"rows": rows, "bases": bases, "fi_ids": fi_ids, "variants": variants, "drops": dict(drops), "configs": [cfg_name(*c) for c in configs],
              "fidelity": fidelity(rows, bases)}

    if debate_agent is not None and debate_budget > 0 and debate_config in result["configs"]:
        result["debate"] = debate_sample(variants, rows, stores, debate_agent, debate_budget, rng, config=debate_config,
                                         bases=bases, controls_budget=(debate_controls_budget if debate_config != PRIMARY else 0))
    return result


def fidelity(rows: list[dict], bases: list[dict]) -> dict:
    """Does re-scoring an UNcorrupted claim (with the reconstructed excerpt) reproduce the original pipeline's score and verdict?"""
    ctrl = {r["base_id"]: r for r in rows if r["config"] == PRIMARY and r["kind"] == "control"}
    n = same = score_ok = verdict_ok = 0
    bad = []
    for b in bases:
        if b["orig_score"] is None or b["claim_id"] not in ctrl:
            continue
        n += 1
        r = ctrl[b["claim_id"]]
        s_ok = abs(r["score"] - float(b["orig_score"])) <= 0.005
        v_ok = b["orig_verdict"] is None or r["verdict"] == b["orig_verdict"]
        score_ok += s_ok
        verdict_ok += v_ok
        if s_ok and v_ok:
            same += 1
        elif len(bad) < 8:
            bad.append((b["claim_id"], b["category"], round(float(b["orig_score"]), 3), round(r["score"], 3), b["orig_verdict"], r["verdict"]))
    return {"n": n, "reproduced": same, "score_ok": score_ok, "verdict_ok": verdict_ok, "mismatches": bad,
            "rate": same / n if n else float("nan")}


def _agents_for(config: str, stores: dict, time_periods) -> dict:
    from verifyfin.agents.verification_agent import VerificationAgent
    variant, kb = config.split("|")
    return {tp: VerificationAgent(vector_store=make_kb_view(stores[tp], kb), **VERIFIER_VARIANTS[variant]) for tp in time_periods}


def debate_sample(variants: list[dict], rows: list[dict], stores: dict, debate_agent, budget: int, rng: random.Random,
                  config: str = PRIMARY, bases: Optional[list[dict]] = None, controls_budget: int = 0) -> dict:
    """
    Run the Debate Agent on a random sample of the CHALLENGED corrupted variants of `config`.
    If controls_budget > 0, also debate the CHALLENGED UNcorrupted originals of that config, to measure how
    many correct claims the full verification + Debate pipeline would still remove (the production
    configuration already has real Debate outcomes from the original run, so it does not need this).
    """
    challenged = [r for r in rows if r["config"] == config and r["kind"] == "corrupted" and r["verdict"] == "challenged"]
    key = lambda v: (v["base_id"], v["operator"], v["mode"])
    by_key = {key(v): v for v in variants}
    sample = rng.sample(challenged, min(budget, len(challenged)))
    agents = _agents_for(config, stores, {r["time_period"] for r in sample})
    out = []
    for r in sample:
        v = by_key[(r["base_id"], r["operator"], r["mode"])]
        c = make_claim(v)
        agents[v["time_period"]].verify_claim(c)
        debate_agent.debate_claim(c)
        out.append({"family": v["family"], "mode": v["mode"], "operator": v["operator"], "resolution": c.debate_resolution})
    rejected = sum(1 for o in out if o["resolution"] == "rejected")
    res = {"config": config, "n_challenged_total": len(challenged), "n_sampled": len(out), "rejected": rejected,
           "upheld": len(out) - rejected, "items": out}

    if controls_budget > 0 and bases:
        base_by_id = {b["claim_id"]: b for b in bases}
        ctl = [r for r in rows if r["config"] == config and r["kind"] == "control" and r["in_fi"] and r["verdict"] == "challenged"]
        picked = rng.sample(ctl, min(controls_budget, len(ctl)))
        c_agents = _agents_for(config, stores, {r["time_period"] for r in picked})
        removed = 0
        for r in picked:
            b = base_by_id[r["base_id"]]
            c = make_claim(b)
            c_agents[b["time_period"]].verify_claim(c)
            debate_agent.debate_claim(c)
            removed += c.debate_resolution == "rejected"
        n_ctl_all = sum(1 for r in rows if r["config"] == config and r["kind"] == "control" and r["in_fi"])
        res["controls"] = {"n_challenged_total": len(ctl), "n_sampled": len(picked), "removed": removed, "n_controls": n_ctl_all}
    return res


# ------------------------------------------------------------------ report

def pct(x: float) -> str:
    return "n/a" if x != x else f"{x:.1%}"


def human_agreement(rows: list[dict], config: str) -> dict:
    """Verification-only agreement with the human labels on ALL numerical+macro claims (usable = routed to GROUNDED)."""
    ctrl = [r for r in rows if r["config"] == config and r["kind"] == "control"]
    tp = sum(1 for r in ctrl if r["human_supported"] and r["verdict"] == "grounded")
    fn = sum(1 for r in ctrl if r["human_supported"] and r["verdict"] != "grounded")
    fp = sum(1 for r in ctrl if not r["human_supported"] and r["verdict"] == "grounded")
    tn = sum(1 for r in ctrl if not r["human_supported"] and r["verdict"] != "grounded")
    n = tp + fp + fn + tn
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": (tp + tn) / n if n else float("nan"),
            "precision": tp / (tp + fp) if tp + fp else float("nan"), "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "kappa": cohens_kappa(tp, fp, fn, tn)}


def format_report(res: dict, n_boot: int = 1500) -> str:
    L: list[str] = []
    add = L.append
    rows = res["rows"]
    prim_cor = [r for r in rows if r["config"] == PRIMARY and r["kind"] == "corrupted"]
    prim_ctl = [r for r in rows if r["config"] == PRIMARY and r["kind"] == "control" and r["in_fi"]]

    add("=" * 78 + "\nFAULT-INJECTION EXPERIMENT (ground truth by construction)\n" + "=" * 78)
    add(f"\nBase claims (human-labelled numerical+macro): {len(res['bases'])}; corrupted from the {len(res['fi_ids'])} judged SUPPORTED.")
    add(f"Corrupted variants: {len(res['variants'])}; dropped (not applicable / not provably false / unvalidated): {res['drops']}")
    fid = res["fidelity"]
    add(f"\nRECONSTRUCTION FIDELITY (do uncorrupted claims re-score exactly as in the original pipeline run?)")
    add(f"  {fid['reproduced']}/{fid['n']} = {pct(fid['rate'])} reproduced both score (+/-0.005) and verdict.")
    if fid["rate"] == fid["rate"] and fid["rate"] < 0.9:
        add("  WARNING: below 90%. The reconstructed excerpts or the knowledge base differ from the original run; interpret with care.")
    for m in fid["mismatches"]:
        add(f"    mismatch {m[0]} [{m[1]}]: original score {m[2]} ({m[4]}) vs now {m[3]} ({m[5]})")

    add("\n" + "-" * 78 + f"\nPRIMARY RESULT: verifier '{PRIMARY}' (current production configuration)\n" + "-" * 78)
    add("A claim is CAUGHT at the verification stage if it is not routed to GROUNDED (BLOCKED, or CHALLENGED -> Debate).")
    add("GROUNDED corrupted claims are certain misses: they would reach the report.\n")
    add(f"  {'category':<10}{'operator':<19}{'mode':<9}{'n':>5}{'GROUNDED':>9}{'CHALL.':>8}{'BLOCKED':>9}   flag rate [95% cluster CI]")
    for s in fi.summarize(prim_cor, ["category", "operator", "mode"], n_boot):
        add(f"  {s['category']:<10}{s['operator']:<19}{s['mode']:<9}{s['n']:>5}{s['grounded']:>9}{s['challenged']:>8}{s['blocked']:>9}   "
            f"{pct(s['flag_rate'])} [{pct(s['cluster_lo'])}, {pct(s['cluster_hi'])}]")
    add("\n  By fault family and injection mode:")
    add(f"  {'family':<13}{'mode':<9}{'n':>6}{'missed (GROUNDED)':>19}   flag rate [95% cluster CI]")
    for s in fi.summarize(prim_cor, ["family", "mode"], n_boot):
        add(f"  {s['family']:<13}{s['mode']:<9}{s['n']:>6}{s['grounded']:>19}   {pct(s['flag_rate'])} [{pct(s['cluster_lo'])}, {pct(s['cluster_hi'])}]")
    for s in fi.summarize(prim_cor, ["category"], n_boot):
        add(f"  ALL {s['category']:<9}{'':<9}{s['n']:>6}{s['grounded']:>19}   {pct(s['flag_rate'])} [{pct(s['cluster_lo'])}, {pct(s['cluster_hi'])}]")
    allc = fi.summarize([dict(r, all="all") for r in prim_cor], ["all"], n_boot)[0] if prim_cor else None
    if allc:
        add(f"  ALL corrupted      {'':<9}{allc['n']:>6}{allc['grounded']:>19}   {pct(allc['flag_rate'])} [{pct(allc['cluster_lo'])}, {pct(allc['cluster_hi'])}]")

    add("\n  FALSE-FLAG RATE on the UNcorrupted originals (all judged SUPPORTED):")
    cs = fi.summarize([dict(r, all="all") for r in prim_ctl], ["all"], n_boot)[0] if prim_ctl else None
    if cs:
        add(f"    {cs['n']} controls: GROUNDED {cs['grounded']}, CHALLENGED {cs['challenged']}, BLOCKED {cs['blocked']} -> "
            f"flagged {pct(cs['flag_rate'])} [{pct(cs['cluster_lo'])}, {pct(cs['cluster_hi'])}]")
        orig = [r for r in prim_ctl if r["orig_usable"] is not None]
        if orig:
            lost = sum(1 for r in orig if not r["orig_usable"])
            add(f"    After the ORIGINAL Debate outcomes, {lost}/{len(orig)} = {pct(lost / len(orig))} of these correct claims were finally removed.")

    sep_pos = [r["score"] for r in prim_ctl]
    sep_neg = [r["score"] for r in prim_cor]
    clean = {r["base_id"]: r["score"] for r in prim_ctl}
    add("\n  SCORE SEPARATION (does the continuous grounding score tell clean from corrupted?)")
    add(f"    AUROC, clean vs corrupted scores: {fi.auroc(sep_pos, sep_neg):.3f}   (0.5 = chance, 1.0 = perfect)")
    pr = fi.paired_clean_higher([(clean[r["base_id"]], r["score"]) for r in prim_cor if r["base_id"] in clean])
    add(f"    Paired (clean claim vs its own corrupted version): clean scored higher {pct(pr['higher'])}, equal {pct(pr['equal'])}, "
        f"lower {pct(pr['lower'])}; mean score drop {pr['mean_drop']:+.3f}")
    for fam in sorted({r["family"] for r in prim_cor}):
        for mode in sorted({r["mode"] for r in prim_cor if r["family"] == fam}):
            sub = [r for r in prim_cor if r["family"] == fam and r["mode"] == mode]
            p2 = fi.paired_clean_higher([(clean[r["base_id"]], r["score"]) for r in sub if r["base_id"] in clean])
            add(f"      {fam:<12}{mode:<9} clean higher {pct(p2['higher']):>6}, equal {pct(p2['equal']):>6}, mean drop {p2['mean_drop']:+.3f}")

    add("\n  GROUNDED-THRESHOLD SWEEP (flag = score below threshold; production = 0.75)")
    add(f"    {'threshold':>9}{'detection':>11}{'false-flag':>12}{'Youden J':>10}")
    for t in fi.threshold_sweep(sep_pos, sep_neg, [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]):
        add(f"    {t['threshold']:>9.2f}{pct(t['detection']):>11}{pct(t['false_flag']):>12}{t['youden_j']:>+10.3f}")

    if "debate" in res:
        d = res["debate"]
        dcfg = d.get("config", PRIMARY)
        d_cor = [r for r in rows if r["config"] == dcfg and r["kind"] == "corrupted"]
        add("\n" + "-" * 78 + f"\nDEBATE-STAGE SAMPLE (configuration '{dcfg}')\n" + "-" * 78)
        wl, wh = fi.wilson(d["rejected"], d["n_sampled"])
        add(f"  {d['n_sampled']} of {d['n_challenged_total']} CHALLENGED corrupted claims were debated: "
            f"{d['rejected']} rejected, {d['upheld']} upheld -> rejection rate {pct(d['rejected'] / d['n_sampled'] if d['n_sampled'] else float('nan'))} "
            f"[Wilson {pct(wl)}, {pct(wh)}]")
        if d["n_sampled"] and d_cor:
            n_all = len(d_cor)
            blocked = sum(1 for r in d_cor if r["verdict"] == "blocked")
            chall = sum(1 for r in d_cor if r["verdict"] == "challenged")
            r_rate = d["rejected"] / d["n_sampled"]
            add(f"  END-TO-END detection estimate (VERIFY-FIN = verification + Debate): blocked {blocked} + challenged {chall} x {pct(r_rate)} "
                f"= {pct((blocked + chall * r_rate) / n_all)} of {n_all} corrupted claims")
            add(f"    bounds: certain misses (GROUNDED) {pct(1 - (blocked + chall) / n_all)}; maximum detection {pct((blocked + chall) / n_all)}")
        if "controls" in d:
            c = d["controls"]
            if c["n_sampled"]:
                rate = c["removed"] / c["n_sampled"]
                est = (c["n_challenged_total"] * rate) / c["n_controls"] if c["n_controls"] else float("nan")
                add(f"  CORRECT claims: {c['n_challenged_total']}/{c['n_controls']} were CHALLENGED; {c['n_sampled']} debated, "
                    f"{c['removed']} wrongly rejected -> estimated end-to-end false-removal {pct(est)} of correct claims")
            else:
                add(f"  CORRECT claims: none were CHALLENGED under '{dcfg}' (0/{c['n_controls']}).")

    add("\n" + "=" * 78 + "\nABLATION: every verifier / knowledge-base configuration on the SAME claims\n" + "=" * 78)
    add("Detection = share of corrupted claims not routed to GROUNDED. False-flag = share of originals not routed to GROUNDED.")
    add("Human agreement = verification-only (usable = GROUNDED) against your labels on all numerical+macro claims.\n")
    add(f"  {'configuration':<22}{'detect':>8}{'(source)':>9}{'(phrasing)':>11}{'(label)':>9}{'false-flag':>12}   human: acc   recall   kappa   FP  FN")
    for cfg in res["configs"]:
        cor = [r for r in rows if r["config"] == cfg and r["kind"] == "corrupted"]
        ctl = [r for r in rows if r["config"] == cfg and r["kind"] == "control" and r["in_fi"]]
        det = lambda rs: sum(1 for r in rs if r["verdict"] != "grounded") / len(rs) if rs else float("nan")
        ha = human_agreement(rows, cfg)
        add(f"  {cfg:<22}{pct(det(cor)):>8}{pct(det([r for r in cor if r['mode'] == 'source'])):>9}"
            f"{pct(det([r for r in cor if r['mode'] == 'phrasing'])):>11}{pct(det([r for r in cor if r['mode'] == 'label'])):>9}"
            f"{pct(det(ctl)):>12}   {ha['accuracy']:>16.3f}{ha['recall']:>8.3f}{ha['kappa']:>8.3f}{ha['fp']:>5}{ha['fn']:>4}")
    add("\n  Detection by fault family (rows) for each configuration:")
    fams = sorted({r["family"] for r in prim_cor})
    add(f"  {'configuration':<22}" + "".join(f"{f:>13}" for f in fams))
    for cfg in res["configs"]:
        cor = [r for r in rows if r["config"] == cfg and r["kind"] == "corrupted"]
        cells = []
        for f in fams:
            sub = [r for r in cor if r["family"] == f]
            cells.append(pct(sum(1 for r in sub if r["verdict"] != "grounded") / len(sub)) if sub else "n/a")
        add(f"  {cfg:<22}" + "".join(f"{c:>13}" for c in cells))
    add("\nREADING THE ABLATION: a mechanism earns its place only if removing it LOWERS detection (or raises the false-flag rate or hurts")
    add("human agreement). Where removal changes nothing, the mechanism is not doing measurable work on this data.")
    return "\n".join(L)


def save_rows_csv(rows: list[dict], path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
