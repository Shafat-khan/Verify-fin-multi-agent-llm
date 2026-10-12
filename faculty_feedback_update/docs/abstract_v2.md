# Abstract (revised draft v2)

**Status:** the numbers in plain text are already established from your pasted outputs. Anything in
**[BRACKETS]** is a placeholder for an experiment that has not been run yet. Fill each one from the
script named in the table below, then delete this note. Do not submit with brackets.

---

## Abstract

Large language models (LLMs) used for financial analysis can produce fluent but unsupported claims,
and in multi-agent pipelines such claims propagate between agents before any check is applied. This
dissertation presents VERIFY-FIN, a multi-agent framework that verifies each financial claim before it
can reach the final report. Analyst agents emit typed claims with mandatory source references;
numerical and macroeconomic values are retrieved by code from SEC XBRL filings and the Federal Reserve's
FRED database and only phrased by the model. A verification agent scores every claim against a
scenario-specific knowledge base, using cosine similarity refined by exact-number and date matching; a
second LLM call adjudicates borderline claims; and only claims that pass are used in synthesis.

VERIFY-FIN was evaluated on 14 company-quarter scenarios (five companies, 171 claims) against a
single-agent baseline with no evidence, **[a single-agent baseline given the same retrieved evidence,]**
and an unverified multi-agent baseline. Because a verifier that both gates and measures the outcome
cannot validate itself, three independent checks were used: a blind human review of all 171 claims; a
fault-injection experiment in which **[N]** provably false corruptions of correct claims were injected
and **[X%, 95% CI …]** were flagged, against a false-flag rate of **[Y%]** on the uncorrupted originals;
and an ablation of each verification mechanism.

In the human review, the unsupported share of claims in the final report fell from 7.0% to 1.9%
(difference 5.1 percentage points; 95% cluster-bootstrap CI 0.6 to 10.9), because 9 of 12 unsupported
claims were removed at the cost of 5 of 159 supported claims (3.1%). All 101 numerical claims were
correct. The adjudication stage recovered 26 correct claims without admitting an error: verification
alone gave accuracy 0.801, the full pipeline 0.953. **[Fault injection showed that faults in the cited
value were caught at …, whereas faults in the sentence alone, wrong line items and wrong dates were
caught at …; removing … reduced detection by …]**

Aggregate unsupported rates did not differ significantly from the unverified multi-agent baseline
(8.5% versus 9.1%; exact Wilcoxon p = **[0.375, confirm with audit_metrics.py]**, only three informative
scenario pairs), as expected when both share the same grounded analyst agents; the benefit lies in the
purity of the report rather than in the raw claim error rate. **[Against the single-agent baseline
given the same evidence, the unsupported rate was …, isolating the contribution of the verification
pipeline from that of evidence access.]** Limitations are a single annotator, five companies, repeated
claims (126 distinct claims among 171), and macro claims that can cite an observation dated after the
period end. The results indicate that claim-level verification can raise report reliability at a small,
measurable cost in discarded correct claims, and that attribution and temporal errors define the main
blind spots of retrieval-based verification.

**Keywords:** large language models; hallucination; retrieval-augmented generation; claim verification;
multi-agent systems; financial analysis.

---

## Where every number in the abstract comes from

| Statement in the abstract | Value | Source (script and output block) |
|---|---|---|
| 14 scenarios, five companies, 171 claims | 14 / 5 / 171 | `check_missing.py`; `analyze_results.py` verdict block |
| Unsupported share of report 7.0% → 1.9% | 7.0%, 1.9% | `score_human_eval.py` "RESEARCH QUESTION" |
| Difference 5.1 pp, CI 0.6 to 10.9 | 5.1; 0.6–10.9 | `score_human_eval.py` "CLUSTER-AWARE UNCERTAINTY", row "Reduction"; cross-checked by `audit_metrics.py` |
| 9 of 12 removed; 5 of 159 lost (3.1%) | 9/12; 5/159 | `score_human_eval.py` "WHAT THE FILTER BUYS AND WHAT IT COSTS" |
| All 101 numerical claims correct | 101/101 | `score_human_eval.py` "BY CLAIM CATEGORY" |
| Debate recovered 26 correct claims; accuracy 0.801 vs 0.953 | 26/26; 0.801; 0.953 | `score_human_eval.py` "ABLATION: Verification only…" |
| 126 distinct claims among 171 | 126 | `score_human_eval.py` cluster block; `audit_metrics.py` |
| 8.5% vs 9.1% | 8.50%, 9.10% | `analyze_results.py` FINAL SUMMARY TABLE |
| Exact Wilcoxon p, 3 informative pairs | **0.375 (to confirm)**, 3 | `analyze_results.py` "Paired comparisons"; `audit_metrics.py` |
| Fault-injection N, detection, false-flag | **[pending]** | `run_fault_injection.py` PRIMARY RESULT |
| Which faults are caught / missed | **[pending]** | `run_fault_injection.py` family × mode table |
| Effect of removing each mechanism | **[pending]** | `run_fault_injection.py` ABLATION table |
| Single-agent + RAG unsupported rate | **[pending]** | `analyze_results.py` (needs `run_evaluation.py --fill-missing --with-rag-baseline`) |

## Rules for finishing it honestly

1. If fault-injection **reconstruction fidelity** is below 90%, do not quote fault-injection numbers
   in the abstract; say the experiment is reported in Chapter 5 with that caveat.
2. If a mechanism's removal changes nothing in the ablation, say so; do not list it as a contribution.
3. If the RAG baseline's unsupported rate is close to VERIFY-FIN's, the abstract must say the
   improvement over the no-evidence baseline is mostly due to evidence access, not verification.
4. Keep "unsupported" (not "hallucinated") for the knowledge-base-relative measure throughout.
5. Never write "reduces hallucination from 99% to 10%" or any variant of it.
