# Response to faculty feedback: what changed, how to run it, how to report it

**Status of this document.** The code described here was written and unit-tested (about 200 tests) in a
sandbox with a pytest-compatible stand-in. It has **not** been run on the real data. Every
"expected" statement below is a hypothesis to be tested, not a result. Replace this paragraph with real
numbers once the experiments have been run.

---

## Run order

| # | Command (project root, venv active) | API tokens | What you get |
|---|---|---|---|
| 0 | `pytest tests/ -v` | none | confirms the install; expect all green |
| 1 | `python scripts/audit_metrics.py` | none | integrity checks + independent recomputation of every headline metric |
| 2 | `python scripts/analyze_results.py` | none | corrected paired tests, explicit sign convention, SD/SE/CI |
| 3 | `python scripts/run_fault_injection.py --dry-run` | none | size of the experiment, nothing executed |
| 4 | `python scripts/run_fault_injection.py --quick 30` | none | smoke test on 30 claims (a few minutes) |
| 5 | `python scripts/run_fault_injection.py` | none | the full fault-injection + ablation experiment |
| 6 | `python scripts/run_fault_injection.py --debate-budget 60` | ~27,000 | adds an end-to-end (verification + Debate) detection estimate |
| 7 | `python scripts/run_evaluation.py --fill-missing --with-rag-baseline` | ~90,000-140,000 (my estimate, not measured) | adds the Single-Agent + RAG condition; resumable |
| 8 | `python scripts/analyze_results.py` again | none | four-condition comparison |

Do **not** delete `data/chroma_store` between the evaluation and steps 5-7: they reuse the scenario knowledge bases.
Run step 7 on a fresh daily Groq quota and not on the same day as step 6.

---

## Point 1: Evaluation is unclear (the verifier is both the component under test and the instrument)

**Valid.** The post-hoc rates use VERIFY-FIN's own Verification + Debate to measure the baselines, and the
natural claims contain almost no errors (12 of 171) for the verifier to detect.

**What changed.** `scripts/run_fault_injection.py` (engine in `verifyfin/evaluation/fault_injection.py`):
- Takes numerical and macro claims you judged SUPPORTED and corrupts them with controlled operators:
  value changes (+1%, +5%, +25%, x10), digit transposition and random digit noise, wrong line item
  (revenue to net income, basic to diluted EPS), wrong period (year shifted), wrong date (+1 month, -1 year),
  wrong series (unemployment vs fed funds).
- Ground truth is **by construction**: a corrupted claim is false, an original is true.
- **Validity guard** (`fault_validators.py`): attribution and temporal corruptions are kept only if XBRL facts
  or the FRED observations in the knowledge base prove them false. This removes accidental truths (a flat
  series, equal basic/diluted EPS, a repeated value).
- **Two modes**: *source* (cited value/date is wrong) and *phrasing* (sentence wrong, excerpt still true). They test
  different things because the verifier takes its expected number from the excerpt.
- **Fidelity check**: uncorrupted claims are re-scored and compared with the original run. Below 90% reproduced,
  the report says so.
- Reports: detection rate per operator/family/mode with **cluster-aware** CIs, false-flag rate on the originals,
  AUROC of the grounding score (clean vs corrupted), a threshold sweep, and optionally a Debate-stage sample.

**How to report.** State the operator list as fixed *before* the first run. Report detection and false-flag
rates together. Say plainly which fault classes are missed. Do not describe fault-injection results as the
real-world hallucination rate: they show the verifier's behaviour on controlled faults.

**Limits.** Synthetic faults are not natural hallucinations (construct validity). Sentiment claims are excluded
(interpretive; no excerpt-based check). The source excerpt is reconstructed from the claim text because the
claim records did not store it; the fidelity check is the safeguard.

**Hypotheses to test (not results).** The verifier probably catches wrong *cited values* better than wrong
*sentences*, and probably misses wrong line items and wrong dates on series where the value exists elsewhere.

---

## Point 2: Cross-verify the metrics

**Valid, and it found two real errors.**

| Metric | Was reported | Correct | Why |
|---|---|---|---|
| Wilcoxon one-sided p (VERIFY-FIN vs Unverified) | 0.2071 | **0.375** (exact permutation; to be confirmed by the audit) | SciPy's default uses a normal approximation when there are zeros/ties; with 3 non-zero pairs it is anti-conservative |
| "Rank-biserial" effect size | 0.333 | **0.5** | 0.333 was (n+ - n-)/n, a sign-count quantity; the rank-biserial uses rank sums |

Neither changes the conclusion (not significant); both would have been wrong in the thesis.

**Also fixed:** Wilson z rounded to 1.96 (now 1.959964); a crash in `analyze_results.py` when a results file lacks
`posthoc_extraction_failed`.

**What changed.** `verifyfin/evaluation/stats_utils.py` (exact sign-flip Wilcoxon, rank-biserial, SD/SE/CI, paired
summaries; cross-tested against SciPy and scikit-learn) and `scripts/audit_metrics.py`, which (A) checks the files
agree with each other, (B) recomputes every headline number with independent code paths, (C) compares with the
figures currently quoted and labels each as PASS / FAIL / WARN.

**How to report.** Quote the exact p-value and the rank-biserial correlation. Cite Pratt (1959) for zeros and
ties, Kerby (2014) for the effect size.

---

## Point 3: Check the +/- differences

- `8.50% +/- 5.42%` is the **SD across scenarios**, not the uncertainty of the mean. SE = 5.42/sqrt(14) = 1.45%;
  the 95% CI is mean +/- t(13) x SE = +/- 3.13, i.e. [5.37, 11.63]. The summary table now prints SD and CI separately.
- Paired differences are **baseline minus VERIFY-FIN**; **positive = VERIFY-FIN lower (better)**. A per-scenario table
  with signs is printed, with the mean difference, SD, SE, t-CI and bootstrap CI.
- "+5.1%" in the human evaluation is **percentage points** (absolute); the relative reduction is about 73%.

---

## Point 4: Single-Agent has no evidence but is judged against a knowledge base

**Valid.** That baseline never saw the evidence it was checked against, so its 100% mostly measures missing
evidence, not missing verification.

**What changed.** `verifyfin/baselines/single_agent_rag_baseline.py`: one LLM call given the **same retrieved
evidence** from the **same scenario knowledge base**, told to use only that evidence; no structured claims, no
verification, no Debate. Added to the harness as an opt-in condition `single_agent_rag`. The analysis explains what
each comparison isolates:

| Comparison | Isolates |
|---|---|
| Single-Agent vs Single-Agent + RAG | value of **evidence access** |
| Single-Agent + RAG vs VERIFY-FIN | value of the **verification pipeline** |
| Unverified Multi-Agent vs VERIFY-FIN | value of **verification alone** |

Its claims are saved to `rag_claims_<timestamp>.jsonl` (the name deliberately does not match `claims_*.jsonl`, so
the sampler, analysis and audit scripts can never pick it up by mistake).

**How to report.** Present the three comparisons, not "99% to 10%". If the RAG baseline's unsupported rate is close
to VERIFY-FIN's, say that most of the gain over the no-evidence baseline comes from evidence access.

**Limits.** All rates are still produced by the same checker (circularity). Hand-label a sample of the RAG
baseline's claims (`sample_for_human_eval.py` accepts any claims file) if time allows.

---

## Point 5: References and abstract

`docs/references.md`: where to cite what, the full Harvard list (your 22 + new), and six problems found in
Chapter 2. `docs/abstract_v2.md`: the revised abstract with placeholders for pending results and a table tracing
each number to a script. **All references are unverified by me (no internet access): check each one.**

---

## Point 6: Ablation experiments

Built, **not yet run**. `VerificationAgent` now has keyword-only switches (defaults identical to production):
`use_numeric_match`, `use_targeted_search`, `use_date_preference`, `require_date_match`, `numeric_boost`. The
fault-injection run sweeps eight configurations on the same claims:

`full`, `semantic_only` (the proposal's plain cosine), `no_date_pref`, `no_targeted`, `no_boost`, `strict_date`,
and knowledge-base subsets `full|filing_only` and `full|filing_fred`.

For each it reports detection, false-flag rate, detection by fault family, and agreement with your human labels
(accuracy, recall, kappa). **A mechanism earns its place only if removing it lowers detection, raises false flags or
hurts human agreement; where it changes nothing, say so.** The Debate Agent ablation already computed from your
labels is accuracy 0.801 (verification only) vs 0.953 (full).

---

## What I could not do

- Run any of the new experiments on your data (no API key, no access to your files).
- Verify the references against their sources.
- Confirm the exact p-value 0.375 from your raw file (derived from the structure of your printed output; the audit
  script will confirm it).
