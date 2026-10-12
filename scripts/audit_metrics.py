"""
Independent audit of every headline metric (faculty feedback point 2:
"cross-verify the metrics carefully"; point 3: "check the +/- differences").

What it does, in three layers:

  A. DATA INTEGRITY. Do the files agree with each other? (claim counts per
     scenario in the results file vs the per-claim file, worksheet vs answer
     key, verdict buckets vs is_usable, labels valid, ids unique ...)

  B. INDEPENDENT RECOMPUTATION. Every headline number is recomputed from the
     RAW files using different code paths from the project's own scripts
     (scikit-learn for accuracy/precision/recall/F1/kappa, scipy for Wilson
     and McNemar, brute-force sign-flip enumeration for the Wilcoxon test, a
     vectorised NumPy cluster bootstrap with a different RNG). It also calls
     the project's own functions and checks they agree with the independent
     versions.

  C. REPORTED vs RECOMPUTED. The figures currently quoted in your notes are
     compared with the recomputed ones. Known issues are labelled, e.g. the
     normal-approximation Wilcoxon p-value (0.2071 vs exact 0.375) and the
     effect size that was computed from sign counts (0.333 vs rank-biserial 0.5).

Sign convention for paired differences: difference = rate(baseline) -
rate(VERIFY-FIN); positive = VERIFY-FIN lower (better).

Usage (from the project root, venv active):
    python scripts/audit_metrics.py [--results PATH] [--claims PATH]
           [--worksheet PATH] [--answer-key PATH]
           [--exclude-cost AAPL-2026-Q2:verify_fin MSFT-2025-Q4:single_agent]
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import math
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np

# ---- figures currently quoted in the project notes (from your pasted outputs) ----
REPORTED = {
    "vf_scenario_mean": (0.0850, 6e-4), "vf_scenario_sd": (0.0542, 6e-4), "vf_ci_lo": (0.0537, 6e-4), "vf_ci_hi": (0.1163, 6e-4),
    "unverified_mean": (0.0910, 6e-4), "unverified_sd": (0.0584, 6e-4),
    "accuracy": (0.953, 6e-4), "precision": (0.981, 6e-4), "recall": (0.969, 6e-4), "f1": (0.975, 6e-4), "kappa": (0.667, 6e-4),
    "unsupported_all": (0.070, 6e-4), "unsupported_kept": (0.019, 6e-4), "reduction": (0.051, 6e-4),
    "mcnemar_p": (0.424, 6e-4), "ablation_accuracy_no_debate": (0.801, 6e-4), "ablation_kappa_no_debate": (0.267, 6e-4),
    "wilcoxon_p_reported": (0.2071, 6e-4), "effect_size_reported": (0.333, 6e-4),
    "cluster_acc_lo": (0.905, 0.012), "cluster_acc_hi": (0.993, 0.012),
    "cluster_reduction_lo": (0.006, 0.012), "cluster_reduction_hi": (0.109, 0.012),
}

VALID_LABELS = {"SUPPORTED", "CONTRADICTED", "UNSUPPORTED", "UNVERIFIABLE"}


class Audit:
    def __init__(self):
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))

    def check(self, ok: bool, name: str, detail: str = "") -> None:
        self.add("PASS" if ok else "FAIL", name, detail)

    def close(self, a: float, b: float, tol: float, name: str, detail: str = "") -> None:
        ok = (a == a and b == b) and abs(a - b) <= tol
        self.add("PASS" if ok else "FAIL", name, f"{detail} [{a:.6g} vs {b:.6g}, |diff|={abs(a - b):.2g}, tol={tol:g}]")

    def summary(self) -> dict:
        c = Counter(s for s, _, _ in self.rows)
        return {"PASS": c.get("PASS", 0), "FAIL": c.get("FAIL", 0), "WARN": c.get("WARN", 0), "INFO": c.get("INFO", 0)}

    def print(self) -> None:
        for status, name, detail in self.rows:
            print(f"  [{status:<4}] {name}" + (f"  --  {detail}" if detail else ""))


# =============================== loading ===============================

def _latest(pattern: str) -> Optional[str]:
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def load_inputs(results: Optional[str], claims: Optional[str], worksheet: Optional[str], answer_key: Optional[str]) -> dict:
    results = results or _latest("outputs/results/evaluation_results_*.json")
    if results is None:
        raise SystemExit("No evaluation_results_*.json found; pass --results.")
    ts = Path(results).stem.replace("evaluation_results_", "")
    claims = claims or (f"outputs/results/claims_{ts}.jsonl" if Path(f"outputs/results/claims_{ts}.jsonl").exists() else _latest("outputs/results/claims_*.jsonl"))
    worksheet = worksheet or "outputs/results/human_eval_worksheet.csv"
    answer_key = answer_key or "outputs/results/human_eval_answer_key.json"
    data = {"paths": {"results": results, "claims": claims, "worksheet": worksheet, "answer_key": answer_key}}
    data["results"] = json.load(open(results))
    data["claims"] = [json.loads(l) for l in open(claims) if l.strip()] if claims and Path(claims).exists() else []
    data["worksheet"] = list(csv.DictReader(open(worksheet, newline=""))) if Path(worksheet).exists() else []
    data["answer_key"] = json.load(open(answer_key)) if Path(answer_key).exists() else {}
    return data


def _fair_rate(r: dict) -> float:
    """Unsupported rate for one results row, computed independently of analyze_results.py."""
    if r["condition"] == "verify_fin":
        t = r.get("total_claims") or 0
        return 1.0 - r["usable_claims_count"] / t if t else float("nan")
    pt = r.get("posthoc_total_claims")
    if pt:
        return 1.0 - r["posthoc_usable_count"] / pt
    return float("nan")


# =============================== A. integrity ===============================

def check_integrity(d: dict, a: Audit) -> None:
    res = d["results"]
    by_cond = defaultdict(set)
    for r in res:
        by_cond[r["condition"]].add(r["time_period"])
    scen_sets = {c: s for c, s in by_cond.items()}
    union = set().union(*scen_sets.values()) if scen_sets else set()
    for c, s in sorted(scen_sets.items()):
        a.check(s == union, f"results: condition '{c}' covers all {len(union)} scenarios", f"missing={sorted(union - s)}")
    dup = Counter((r["time_period"], r["condition"]) for r in res)
    a.check(all(v == 1 for v in dup.values()), "results: exactly one row per (scenario, condition)",
            f"duplicates={[k for k, v in dup.items() if v > 1]}")

    if d["claims"]:
        per = defaultdict(lambda: [0, 0])
        for c in d["claims"]:
            per[c["time_period"]][0] += 1
            per[c["time_period"]][1] += 1 if c["is_usable"] else 0
        for r in sorted((r for r in res if r["condition"] == "verify_fin"), key=lambda r: r["time_period"]):
            tp = r["time_period"]
            a.check(per[tp][0] == r["total_claims"] and per[tp][1] == r["usable_claims_count"],
                    f"results vs claims file agree for {tp}",
                    f"results total/usable={r['total_claims']}/{r['usable_claims_count']}, claims file={per[tp][0]}/{per[tp][1]}")
        for c in d["claims"]:
            v, dr = (c.get("verdict") or "").lower(), c.get("debate_resolution")
            expect = v == "grounded" or (v == "challenged" and dr == "upheld")
            if bool(c["is_usable"]) != expect:
                a.add("FAIL", "claims file: is_usable consistent with verdict/debate", f"{c.get('claim_id')}: verdict={v} debate={dr} is_usable={c['is_usable']}")
                break
        else:
            a.add("PASS", "claims file: is_usable consistent with verdict/debate_resolution for every claim")

    ws, key = d["worksheet"], d["answer_key"]
    if ws:
        ids = [r["claim_id"] for r in ws]
        a.check(len(ids) == len(set(ids)), "worksheet: claim_ids unique")
        a.check(set(ids) == set(key), "worksheet ids == answer-key ids", f"only_ws={len(set(ids) - set(key))}, only_key={len(set(key) - set(ids))}")
        labels = [(r.get("human_label") or "").strip().upper() for r in ws]
        a.check(all(l in VALID_LABELS for l in labels), "worksheet: every row has a valid label",
                f"blank={sum(1 for l in labels if not l)}, invalid={sum(1 for l in labels if l and l not in VALID_LABELS)}")
        if d["claims"]:
            a.check(Counter((r["claim_category"], r["claim_text"].strip()) for r in ws) ==
                    Counter((c["claim_category"], c["claim_text"].strip()) for c in d["claims"]),
                    "worksheet claims == claims file (same category+text multiset)",
                    f"worksheet={len(ws)} rows, claims file={len(d['claims'])} rows")
        bad = [k for k, v in key.items() if bool(v["is_usable"]) != (v["outcome_bucket"] in ("GROUNDED", "UPHELD"))]
        a.check(not bad, "answer key: is_usable consistent with outcome_bucket", f"inconsistent={bad[:3]}")
        a.add("INFO", "claim categories in worksheet", str(dict(Counter(r["claim_category"] for r in ws))))


# =============================== B. recomputation ===============================

def scenario_stats(d: dict, a: Audit) -> dict:
    from scipy import stats

    res = d["results"]
    out = {}
    for cond in sorted({r["condition"] for r in res}):
        rows = [r for r in res if r["condition"] == cond]
        rates = np.array([_fair_rate(r) for r in rows], dtype=float)
        rates = rates[~np.isnan(rates)]
        n = len(rates)
        mean = rates.mean() if n else float("nan")
        sd = rates.std(ddof=1) if n > 1 else float("nan")
        se = sd / math.sqrt(n) if n > 1 else float("nan")
        half = stats.t.ppf(0.975, n - 1) * se if n > 1 else float("nan")
        if cond == "verify_fin":
            pooled = 1 - sum(r["usable_claims_count"] for r in rows) / sum(r["total_claims"] for r in rows)
        else:
            pt = [r for r in rows if r.get("posthoc_total_claims")]
            pooled = 1 - sum(r["posthoc_usable_count"] for r in pt) / sum(r["posthoc_total_claims"] for r in pt) if pt else float("nan")
        out[cond] = dict(n=n, mean=mean, sd=sd, se=se, lo=mean - half, hi=mean + half, pooled=pooled, rates=rates)
        a.add("INFO", f"{cond}: n={n}", f"mean {mean:.4%} | SD {sd:.4%} | SE {se:.4%} | 95% CI [{mean - half:.4%}, {mean + half:.4%}] "
                                        f"(= mean +/- {half:.4%}) | pooled claim-level {pooled:.4%}")
        if n > 1:
            a.close(half, stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n), 1e-12, f"{cond}: CI half-width == t*SD/sqrt(n)")
    return out


def brute_force_signflip_p(diffs: list[float]) -> Optional[dict]:
    """Independent exact sign-flip test by explicit enumeration (only for <= 20 non-zero differences)."""
    from scipy import stats

    nz = np.array([x for x in diffs if x != 0 and x == x])
    m = len(nz)
    if m == 0 or m > 20:
        return None
    # round before ranking: mathematically equal differences (e.g. +1/13 and -1/13) can differ by ~1e-17 of float noise,
    # and scipy.rankdata would then break the tie arbitrarily and change the rank sums
    ranks = stats.rankdata(np.round(np.abs(nz), 10))
    obs = ranks[nz > 0].sum()
    ge = tot = 0
    for signs in product((0, 1), repeat=m):
        t = sum(r for r, s in zip(ranks, signs) if s)
        ge += t >= obs - 1e-9
        tot += 1
    t_plus, t_minus = obs, ranks.sum() - obs
    return {"p_greater": ge / tot, "t_plus": t_plus, "t_minus": t_minus, "rank_biserial": (t_plus - t_minus) / ranks.sum(),
            "sign_net": ((nz > 0).sum() - (nz < 0).sum()) / m, "m": m}


def paired_checks(d: dict, sc: dict, a: Audit) -> dict:
    from scipy import stats

    from verifyfin.evaluation.stats_utils import wilcoxon_signflip

    res = d["results"]
    key = lambda r: (r["ticker"], r["time_period"])
    vf = {key(r): _fair_rate(r) for r in res if r["condition"] == "verify_fin"}
    found = {}
    for cond in ("unverified_multiagent", "single_agent_rag", "single_agent"):
        base = {key(r): _fair_rate(r) for r in res if r["condition"] == cond}
        common = sorted(set(vf) & set(base))
        pairs = [(base[k], vf[k]) for k in common if base[k] == base[k] and vf[k] == vf[k]]
        if len(pairs) < 3:
            continue
        diffs = [b - v for b, v in pairs]            # baseline - VERIFY-FIN
        a.add("INFO", f"paired {cond} vs verify_fin: n={len(pairs)}",
              f"VF lower {sum(x > 0 for x in diffs)}, higher {sum(x < 0 for x in diffs)}, tied {sum(x == 0 for x in diffs)}; "
              f"mean diff {np.mean(diffs):+.4%}, SD {np.std(diffs, ddof=1):.4%}")
        bf = brute_force_signflip_p(diffs)
        ours = wilcoxon_signflip(diffs, "greater")
        if bf:
            a.close(ours["p_value"], bf["p_greater"], 1e-12, f"{cond}: exact Wilcoxon p (DP) == brute-force enumeration")
            a.close(ours["rank_biserial"], bf["rank_biserial"], 1e-12, f"{cond}: rank-biserial from rank sums")
        try:
            sp = stats.wilcoxon([b for b, _ in pairs], [v for _, v in pairs], alternative="greater").pvalue
        except ValueError:
            sp = float("nan")
        gap = abs(sp - ours["p_value"])
        a.add("WARN" if gap > 0.01 else "PASS", f"{cond}: scipy default Wilcoxon p vs exact permutation p",
              f"scipy {sp:.4f} vs exact {ours['p_value']:.4f}" + ("  -> scipy used a normal approximation; report the exact value" if gap > 0.01 else ""))
        found[cond] = {"scipy_p": sp, "exact_p": ours["p_value"], "rank_biserial": ours["rank_biserial"], "sign_net": ours["sign_net_advantage"], "n_nonzero": ours["n_nonzero"]}
    return found


def human_recompute(d: dict, a: Audit) -> Optional[dict]:
    from scipy import stats
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, precision_score, recall_score

    ws, key = d["worksheet"], d["answer_key"]
    if not ws or not key:
        return None
    rows = []
    for r in ws:
        lab = (r.get("human_label") or "").strip().upper()
        if lab not in VALID_LABELS or r["claim_id"] not in key:
            continue
        k = key[r["claim_id"]]
        rows.append({"cat": r["claim_category"], "text": r["claim_text"].strip(), "y": int(lab == "SUPPORTED"),
                     "p": int(bool(k["is_usable"])), "bucket": k["outcome_bucket"]})
    y = np.array([r["y"] for r in rows]); p = np.array([r["p"] for r in rows]); n = len(rows)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    m = dict(n=n, tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
             accuracy=accuracy_score(y, p), precision=precision_score(y, p), recall=recall_score(y, p), f1=f1_score(y, p), kappa=cohen_kappa_score(y, p))
    a.add("INFO", "human evaluation (sklearn)", f"n={n} TP={tp} FP={fp} FN={fn} TN={tn} | acc {m['accuracy']:.4f} prec {m['precision']:.4f} rec {m['recall']:.4f} F1 {m['f1']:.4f} kappa {m['kappa']:.4f}")

    # compare with the project's own scorer
    spec = importlib.util.spec_from_file_location("score_human_eval", Path(__file__).parent / "score_human_eval.py")
    sh = importlib.util.module_from_spec(spec); spec.loader.exec_module(sh)
    joined = sh.validate_and_join(ws, key)
    pm = sh.compute_binary_metrics(joined)
    for name in ("accuracy", "precision", "recall", "f1"):
        a.close(pm[name], m[name], 1e-12, f"scorer vs sklearn: {name}")
    a.close(pm["cohens_kappa"], m["kappa"], 1e-12, "scorer vs sklearn: Cohen's kappa")
    for name, k_, d_ in (("accuracy", tp + tn, n), ("precision", tp, tp + fp), ("recall", tp, tp + fn)):
        ref = stats.binomtest(int(k_), int(d_)).proportion_ci(confidence_level=0.95, method="wilson")
        lo, hi = sh.wilson_interval(int(k_), int(d_))
        a.close(lo, ref.low, 1e-4, f"scorer Wilson low vs scipy: {name}", "tolerance 1e-4 because the scorer rounds z to 1.96")
        a.close(hi, ref.high, 1e-4, f"scorer Wilson high vs scipy: {name}")
    mc = sh.mcnemar_exact_vs_accept_all(joined)
    a.close(mc["p"], stats.binomtest(int(tn), int(tn + fn), 0.5).pvalue, 1e-12, "scorer McNemar p vs scipy binomtest",
            f"discordant: {tn} bad removed vs {fn} good lost")
    n_bad_all, n_kept = int(fp + tn), int(tp + fp)
    m.update(unsupported_all=n_bad_all / n, unsupported_kept=fp / n_kept if n_kept else float("nan"), mcnemar_p=mc["p"])
    m["reduction"] = m["unsupported_all"] - m["unsupported_kept"]
    a.add("INFO", "report purity", f"unsupported share all {m['unsupported_all']:.4%} -> kept {m['unsupported_kept']:.4%} (reduction {m['reduction']:+.4%}); "
                                   f"removed {tn}/{n_bad_all} unsupported, lost {fn}/{tp + fn} supported")

    # Debate counterfactual, independently
    p0 = np.array([int(r["bucket"] == "GROUNDED") for r in rows])
    tn0, fp0, fn0, tp0 = confusion_matrix(y, p0, labels=[0, 1]).ravel()
    m["ablation_accuracy_no_debate"] = accuracy_score(y, p0); m["ablation_kappa_no_debate"] = cohen_kappa_score(y, p0)
    a.add("INFO", "Debate counterfactual (only GROUNDED usable)", f"TP={tp0} FP={fp0} FN={fn0} TN={tn0} | acc {m['ablation_accuracy_no_debate']:.4f} kappa {m['ablation_kappa_no_debate']:.4f}")

    # independent vectorised cluster bootstrap, several seeds
    clusters = defaultdict(lambda: np.zeros(4))
    col = {(1, 1): 0, (0, 1): 1, (1, 0): 2, (0, 0): 3}   # tp, fp, fn, tn
    for r in rows:
        clusters[(r["cat"], r["text"])][col[(r["y"], r["p"])]] += 1
    arr = np.array(list(clusters.values()))   # columns: tp, fp, fn, tn
    K = len(arr)
    a.add("INFO", "clusters (identical category+claim text)", f"{K} distinct claims from {n} rows")
    bounds = {"acc": [], "red": [], "kappa": []}
    for seed in (1, 2, 3, 4, 5):
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, K, size=(5000, K))
        s = arr[idx].sum(axis=1)
        TP, FP, FN, TN = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
        N = TP + FP + FN + TN
        acc = (TP + TN) / N
        with np.errstate(divide="ignore", invalid="ignore"):
            red = (FP + TN) / N - FP / (TP + FP)
            pe = ((TP + FN) / N) * ((TP + FP) / N) + ((FP + TN) / N) * ((FN + TN) / N)
            kap = (acc - pe) / (1 - pe)
        for nm, v in (("acc", acc), ("red", red), ("kappa", kap)):
            bounds[nm].append(tuple(np.nanpercentile(v, [2.5, 97.5])))
    for nm in ("acc", "red", "kappa"):
        los, his = [b[0] for b in bounds[nm]], [b[1] for b in bounds[nm]]
        a.add("INFO", f"cluster bootstrap {nm} (5 seeds)", f"lower {min(los):.3f}..{max(los):.3f}, upper {min(his):.3f}..{max(his):.3f} "
                                                       f"(Monte-Carlo spread {max(max(los) - min(los), max(his) - min(his)):.3f})")
    m["cluster_acc_lo"], m["cluster_acc_hi"] = float(np.mean([b[0] for b in bounds["acc"]])), float(np.mean([b[1] for b in bounds["acc"]]))
    m["cluster_reduction_lo"], m["cluster_reduction_hi"] = float(np.mean([b[0] for b in bounds["red"]])), float(np.mean([b[1] for b in bounds["red"]]))
    return m


def cost_checks(d: dict, a: Audit, exclude: list[str]) -> None:
    res = d["results"]
    ex = {tuple(x.split(":")) for x in exclude}
    for cond in sorted({r["condition"] for r in res}):
        rows = [r for r in res if r["condition"] == cond]
        kept = [r for r in rows if (r["time_period"], cond) not in ex]
        f = lambda rs, k: float(np.mean([r[k] for r in rs])) if rs else float("nan")
        a.add("INFO", f"cost {cond}", f"tokens all rows {f(rows, 'llm_total_tokens'):.1f} | excluding gap-filled {f(kept, 'llm_total_tokens'):.1f} "
                                    f"(n={len(kept)}); API calls {f(rows, 'llm_api_call_count'):.2f} -> {f(kept, 'llm_api_call_count'):.2f}; "
                                    f"median runtime {float(np.median([r['runtime_seconds'] for r in rows])):.1f}s vs mean {f(rows, 'runtime_seconds'):.1f}s")


# =============================== C. reported vs recomputed ===============================

def compare_reported(sc: dict, paired: dict, human: Optional[dict], a: Audit) -> None:
    def cmp(key: str, value: float, label: str) -> None:
        exp, tol = REPORTED[key]
        a.add("PASS" if abs(value - exp) <= tol else "FAIL", f"reported {label}", f"reported {exp} vs recomputed {value:.4f} (tol {tol})")

    if "verify_fin" in sc:
        v = sc["verify_fin"]
        cmp("vf_scenario_mean", v["mean"], "VERIFY-FIN scenario mean"); cmp("vf_scenario_sd", v["sd"], "VERIFY-FIN scenario SD")
        cmp("vf_ci_lo", v["lo"], "VERIFY-FIN 95% CI low"); cmp("vf_ci_hi", v["hi"], "VERIFY-FIN 95% CI high")
    if "unverified_multiagent" in sc:
        cmp("unverified_mean", sc["unverified_multiagent"]["mean"], "Unverified scenario mean")
        cmp("unverified_sd", sc["unverified_multiagent"]["sd"], "Unverified scenario SD")
    if human:
        for k, lab in (("accuracy", "accuracy"), ("precision", "precision"), ("recall", "recall"), ("f1", "F1"), ("kappa", "kappa"),
                       ("unsupported_all", "unsupported share (all)"), ("unsupported_kept", "unsupported share (kept)"), ("reduction", "reduction"),
                       ("mcnemar_p", "McNemar p"), ("ablation_accuracy_no_debate", "no-Debate accuracy"), ("ablation_kappa_no_debate", "no-Debate kappa"),
                       ("cluster_acc_lo", "cluster accuracy CI low"), ("cluster_acc_hi", "cluster accuracy CI high"),
                       ("cluster_reduction_lo", "cluster reduction CI low"), ("cluster_reduction_hi", "cluster reduction CI high")):
            cmp(k, human[k], lab)
    u = paired.get("unverified_multiagent")
    if u:
        exp_p = REPORTED["wilcoxon_p_reported"][0]
        a.add("WARN" if abs(u["exact_p"] - exp_p) > 0.01 else "PASS", "reported Wilcoxon p (VERIFY-FIN vs Unverified)",
              f"reported {exp_p} (scipy normal approximation {u['scipy_p']:.4f}) vs EXACT permutation p {u['exact_p']:.4f} "
              f"({u['n_nonzero']} non-zero pairs)  -> quote the exact value")
        exp_e = REPORTED["effect_size_reported"][0]
        a.add("WARN" if abs(u["rank_biserial"] - exp_e) > 0.01 else "PASS", "reported effect size",
              f"reported {exp_e} was (n+ - n-)/n = net sign advantage {u['sign_net']:.3f}; the rank-biserial correlation from rank sums is {u['rank_biserial']:.3f}")


def run_audit(d: dict, exclude_cost: list[str], reported: bool = True) -> Audit:
    a = Audit()
    a.add("INFO", "files", json.dumps(d["paths"]))
    check_integrity(d, a)
    sc = scenario_stats(d, a)
    paired = paired_checks(d, sc, a)
    human = human_recompute(d, a)
    cost_checks(d, a, exclude_cost)
    if reported:
        compare_reported(sc, paired, human, a)
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results"); ap.add_argument("--claims"); ap.add_argument("--worksheet"); ap.add_argument("--answer-key")
    ap.add_argument("--exclude-cost", nargs="*", default=["AAPL-2026-Q2:verify_fin", "MSFT-2025-Q4:single_agent"],
                    help="scenario:condition pairs whose cost counters are cache-reduced (gap-filled)")
    ap.add_argument("--no-reported", action="store_true", help="skip the comparison with the figures quoted in the notes")
    args = ap.parse_args()
    d = load_inputs(args.results, args.claims, args.worksheet, args.answer_key)
    a = run_audit(d, args.exclude_cost, reported=not args.no_reported)
    print("=" * 78 + "\nMETRICS AUDIT\n" + "=" * 78)
    a.print()
    s = a.summary()
    print("\n" + "=" * 78)
    print(f"PASS {s['PASS']}   FAIL {s['FAIL']}   WARN {s['WARN']}   (INFO lines are context, not checks)")
    print("FAIL = a number or file inconsistency to investigate before it goes in the thesis.")
    print("WARN = a number that is computable but misleading as currently reported (see the line for the fix).")


if __name__ == "__main__":
    main()
