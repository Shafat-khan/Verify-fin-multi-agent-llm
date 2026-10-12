# Stage 2 update: closing the gaps the experiments exposed

Written 11 Oct 2026, after three runs: the metric audit, the fault-injection + ablation run, and the
Single-Agent + RAG baseline. This file explains what each run showed, what was changed in response, what is
still unproven, and exactly what to run next.

## 1. What the three runs showed

| Run | Finding | Consequence |
|---|---|---|
| Audit | 65 checks pass, 0 fail. Exact Wilcoxon p = 0.375 (scipy's 0.2071 was a normal approximation); rank-biserial r = 0.500 (0.333 was the net sign advantage). | Report the exact values. With 3 non-zero pairs the smallest possible exact one-sided p is 0.125, so "VERIFY-FIN vs Unverified" cannot reach significance. |
| Fault injection, 2,159 provably false variants of 153 correct claims | Wrong evidence caught ~88% (magnitude 88.5%, typos 88.6%). Wrong sentence with true evidence caught ~13% (13.3%, 14.2%). Dates ~13% in both modes. Wrong label/series 46.2%. | The verifier does not read the number or date written in the sentence. This is the main weakness found. |
| Debate on 60 corrupted CHALLENGED claims | 60/60 rejected (Wilson lower bound 94%); 22/24 correct challenged claims were upheld in the original run. | Debate is accurate both ways, but it adds no detection beyond the verifier: the 53.5% marked GROUNDED never reach it. |
| Ablation | Numeric match/boost is the core mechanism (without it 86-87% of CORRECT claims are flagged). Targeted search helps. FRED data is needed. Date preference shows no measurable effect. | Keep numeric match and targeted search; do not claim date preference matters. |
| Single-Agent + RAG | Automated unsupported rate 71.4% (n=10 of 14), grounded 6.3%. **4 of 14 reports were EMPTY.** | See sections 2 and 3: neither number can be used as it stands. |

## 2. Two defects found in the RAG run (both mine)

**Empty reports.** For 4 of 14 scenarios the model returned no text. The log shows the post-hoc extractor made no
LLM call at all, which only happens for an empty report. The model is a reasoning model, and its hidden reasoning
tokens count against `max_tokens`; I had set 700. Excluding those scenarios from the mean is a selection effect.
*Fix:* `max_tokens` 2500, one retry at 5000, an explicit `last_report_empty` flag, an error log line, and
`--redo-failed` so failed rows are re-run rather than averaged around. The analysis now names the excluded
scenarios per condition.

**The 71.4% is not a hallucination rate.** The verifier receives each structured claim's source excerpt and can
search for its exact number. A sentence pulled out of free text has no excerpt, so the verifier falls back to
cosine similarity alone, and the ablation measured that mode flagging **86% of correct claims**. So the RAG
baseline (and the no-evidence Single-Agent) is being judged by an instrument that is blind in exactly its case.
This is the same confound the faculty raised in point 1, in a new place.
*Fix:* compare conditions on **human-labelled error left in the report** (`scripts/baseline_human_eval.py`).
For VERIFY-FIN and the Unverified baseline this is already computable from your existing 171 labels.

## 3. What changed

* `verifyfin/agents/text_consistency.py` (new): deterministic check that the number and date written in the
  claim text agree with the cited excerpt. Rounding-aware ("$116 billion" accepted for 116,017,000,000;
  "$117.2 billion" rejected). No LLM, no embeddings.
* `VerificationAgent(check_text_consistency=..., text_mismatch_action="challenge"|"block")`. **Off by default,
  so every earlier result stays reproducible.** A mismatch caps the score just below the GROUNDED threshold, so
  the Debate Agent decides ("challenge"), or below the CHALLENGED threshold ("block").
* Fault runner: new variants `text_check`, `text_check_strict`, `text_check_block`; `--v2`;
  `--debate-config` (the Debate Agent can now be run on any configuration, and on that configuration's
  challenged ORIGINALS to measure false removal end to end).
* `scripts/check_text_on_real_claims.py`: applies the check to real analyst output with real excerpts.
* `scripts/baseline_human_eval.py`: blind worksheet for the RAG baseline + the error-left-in-report table.
* RAG baseline fix, `--redo-failed`, analysis wording and per-condition exclusion notes.

## 4. Predictions, written before running (so the result cannot be bent to fit)

If the diagnosis is right, then with `text_check`:
1. Detection of **phrasing** faults for magnitude and typo rises from ~13-14% to at least 90%.
2. Detection of **temporal** faults in phrasing mode rises from ~12% to at least 80% (the sentence's date is wrong, the excerpt's is right).
3. **Source-mode** detection does not fall (the sentence and the excerpt agree, so the check is silent there).
4. False flags on the 153 correct originals do not rise (they will not, by construction, see the caveat below).
5. Wrong-label faults (attribution) change little: the number is right and only the label is wrong.

If (1) or (2) fail, the diagnosis is wrong or incomplete, and that is a finding to report.

## 5. What this does NOT prove (state these in the thesis)

* **Circularity of the control test.** In the fault-injection experiment the excerpt of a clean claim is rebuilt
  from its own text, so the text check cannot disagree with it. The false-flag rate for `text_check` there is a
  lower bound. The real test is `check_text_on_real_claims.py`, which uses real excerpts.
* Phrasing faults are synthetic. In the real data 153 of 157 claims were supported, so the check's real-world
  benefit is protection against a failure that rarely occurred, not a measured improvement in report accuracy.
* The check validates only the excerpt's single value and date. A right number under a wrong label is not caught.
* Rounded figures accept a corruption smaller than their own rounding step.
* One annotator; development tickers overlap evaluation tickers.

## 6. Commands, in order (project root, venv active)

```bash
export ANONYMIZED_TELEMETRY=False          # hides the harmless ChromaDB telemetry lines
pytest tests/ -q 2>&1 | tail -8            # expect: all pass

# A. does the new check misfire on real claims?  (no knowledge base, normally ~0 tokens)
python scripts/check_text_on_real_claims.py 2>&1 | tail -60

# B. v1 vs v2 on the same 2,159 corruptions, with Debate on the v2 configuration  (~45,000 tokens)
python scripts/run_fault_injection.py --v2 --debate-budget 60 --debate-config "text_check|all"
LATEST=$(ls -t outputs/results/fault_injection_report_*.txt | head -1)
sed -n '/^DEBATE-STAGE/,$p' "$LATEST"

# C. re-run the 4 empty RAG reports  (~12,000 tokens; resumable)
python scripts/run_evaluation.py --fill-missing --with-rag-baseline --redo-failed 2>&1 | tail -30
python scripts/analyze_results.py > outputs/results/analysis_output_rag_fixed.txt 2>&1
tail -45 outputs/results/analysis_output_rag_fixed.txt

# D. human labels for the RAG baseline  (~1 hour of labelling)
python scripts/baseline_human_eval.py export --n 60
#   open labelling_tool.html, load outputs/results/rag_worksheet.csv, label every row,
#   save over the same file, then:
python scripts/baseline_human_eval.py score
```

Run B and C on different days if the Groq daily cap (200,000 tokens) is close.
