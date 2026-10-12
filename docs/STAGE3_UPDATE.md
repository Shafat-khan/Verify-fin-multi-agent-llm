# Stage 3 update: evidence alignment (v3) and the no-look-ahead macro fix

## Why this exists
Human labelling of the final VERIFY-FIN report found 4 of 157 claims that were not supported by the passage they cite
(2.5%, CI 1.0-6.4%). All four were TRUE statements grounded to the WRONG passage, and the correct passage was in the
knowledge base:
* "FFR 3.64 on 2026-04-01" cited the chunk "3.64 on 2026-03-01" (same value, other month); the April chunk existed.
* "unemployment 4.1 on 2026-07-01" cited a bank's forecast table containing "4.1" instead of "UNRATE observation: 4.1 on 2026-07-01".
Cause: the number match only looks inside the top-8 semantic hits and the date preference is only a tie-break inside them.
Fault injection pointed the same way: wrong-date claims were caught ~13% of the time and wrong-label claims ~45%.

A second finding from the same investigation: the Macro Agent used the observation CLOSEST to the period end, which for a
quarter ending 28 March picked the observation dated 1 April (data from after the period), and FRED data was fetched and
indexed up to today. Both are look-ahead.

## What changed
| File | Change |
|---|---|
| `verifyfin/agents/evidence_alignment.py` (new) | pure functions: whole-number matching, FRED record parsing, series and line-item detection, filing-table column/date alignment |
| `verifyfin/agents/verification_agent.py` | four new keyword switches, all default OFF (old results stay reproducible): `date_aligned_search`, `label_check`, `filing_period_check`, `exact_record_grounds` |
| `verifyfin/agents/macro_agent.py` | uses the latest observation dated ON OR BEFORE the target date (never after) |
| `verifyfin/data_pipeline/kb_indexer.py`, `scripts/run_evaluation.py` | new knowledge bases index FRED only up to the period end (`as_of`) |
| `scripts/prune_future_fred.py` (new) | applies the same cutoff to your EXISTING knowledge bases (dry run by default) |
| `verifyfin/evaluation/fault_runner.py`, `scripts/run_fault_injection.py` | configs `v3_date`, `v3_label`, `v3_period`, `v3_aligned`, `v3_exact`; flag `--v3` |
| `scripts/check_v3_on_real_claims.py` (new) | verifies the REAL analyst claims under v1 / v2 / v3 and lists every claim whose outcome changed |

The v3 checks, in plain words
* **date_aligned_search** (FRED claims): look up the exact record "SERIES observation: value on DATE". Found -> cite it. Not found -> the claim cannot be GROUNDED.
* **label_check**: the series (FRED) or line item (filings) the sentence names must match the matched passage. Only a POSITIVE mismatch counts; an unlabelled chunk is "unknown", never a mismatch.
* **filing_period_check**: in a filing table the column holding the number must carry the claimed period-end date. On a mismatch it first tries another chunk, then flags.
* **exact_record_grounds**: a claim confirmed as an exact structured record goes to GROUNDED even if cosine similarity is low (an exact lookup is stronger evidence than cosine). A text or label mismatch always wins.

## Predictions written BEFORE running (so they can be wrong)
Fault injection (same worksheet, same seed), compared with the v2 `text_check|all` row in the same run:
1. `v3_date`: macro date-shift detection rises from ~20% to >= 95% (by construction: a false date has no exact record).
2. `v3_period`: numeric period-shift detection rises from ~9% to between 50% and 90% (depends on how many matched chunks keep their header row).
3. `v3_label`: macro series-swap detection >= 95%; numeric label-swap detection between 70% and 95%.
4. False-flag rate on the clean originals (v2: 15.7%): `v3_aligned` within +1 point of that; `v3_exact` clearly lower.
5. Certain misses (GROUNDED corrupted): v2 12.6% -> v3 <= 5%.
Real claims: zero claims in a "WORSE" group; the 4 known citation errors appear under "BETTER".
If a prediction is wrong, report it as wrong.

## HONEST LIMITS (put these in the thesis)
* v3 was designed AFTER seeing these 4 failures. Its improvement on the development scenarios is therefore optimistic. The honest test is the frozen-code run on NEW tickers.
* Fault-injection detection of date and label faults is close to 100% partly by construction (the validator that proves a variant false uses the same data). The real-world evidence is the real-claims check (false alarms and repaired citations).
* Duration is not checked: a nine-month figure cited as a quarterly one carries the same date and is not caught.
* "Dated on or before the period end" is not "published on or before": a monthly value dated the 1st describes the whole month, and FRED serves revised values. A fully point-in-time analysis would need ALFRED vintages.
* The line-item vocabulary is a fixed English synonym list; an unlisted label gives "unknown".

## Run order (project root, venv active)
```
unzip -o stage3_update.zip
pytest tests/ -q                                   # expect about 491 passed
cp -r data/chroma_store data/chroma_store_backup_before_prune
python scripts/prune_future_fred.py                # dry run: how many future FRED chunks
python scripts/prune_future_fred.py --apply
python scripts/check_v3_on_real_claims.py > outputs/results/v3_real_claims_output.txt 2>&1
python scripts/run_fault_injection.py --v2 --v3 --debate-budget 60 --debate-config "v3_exact|all" > outputs/results/fault_v3_output.txt 2>&1
```
Do NOT delete `data/chroma_store` (the backup copy is only a safety net).
Token cost: `check_v3_on_real_claims.py` re-phrases the macro claims whose observation date changed (about 14 short LLM calls, roughly 10-14k tokens); the fault-injection debate sample is about 45k tokens.
