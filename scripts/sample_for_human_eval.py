"""
Stratified sampling for the human spot-check evaluation (Proposal
§2.5: "Conduct human spot-check evaluation").

Reads the per-claim JSONL produced by run_evaluation.py (VERIFY-FIN's
claims only -- see run_evaluation.py's _save_claim_records docstring)
and draws a stratified sample across:
    - ticker (AAPL, MSFT, JPM, JNJ, XOM)
    - claim_category (numerical / macro / sentiment)
    - outcome (grounded / upheld / rejected / blocked)

producing TWO files:
    1. A BLIND labeling worksheet (CSV) -- claim text and the exact
       evidence VERIFY-FIN retrieved, with NO system verdict shown, so
       labelling is not biased by already knowing the system's answer.
    2. A hidden answer key (JSON) mapping claim_id -> the system's own
       verdict/debate_resolution/is_usable -- do not open this until
       after labelling is complete.

Sampling strategy: at least 1 claim per non-empty stratum (so rare
combinations, e.g. a single ticker's rejected sentiment claim, are
still represented), then remaining budget allocated proportionally to
stratum size, capped at each stratum's actual population and at
TARGET_SAMPLE_SIZE overall. A fixed random seed is used throughout for
reproducibility.

Usage (from the project root, with venv activated):
    python scripts/sample_for_human_eval.py [path/to/claims.jsonl] [--n 180]

If no path is given, the most recent claims_*.jsonl in
outputs/results/ is used automatically.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from verifyfin.utils.io_utils import resolve_path

_DEFAULT_SAMPLE_SIZE = 180
_RANDOM_SEED = 42  # fixed for reproducibility -- report this seed in your Methodology chapter

_HUMAN_LABEL_OPTIONS = "SUPPORTED / CONTRADICTED / UNSUPPORTED / UNVERIFIABLE"


def find_latest_claims_file() -> Path:
    results_dir = resolve_path("outputs/results")
    candidates = sorted(results_dir.glob("claims_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"No claims_*.jsonl files found in {results_dir}. "
            f"Run scripts/run_evaluation.py first (it now persists per-claim records)."
        )
    return candidates[-1]


def load_claims(path: Path) -> list[dict]:
    claims = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                claims.append(json.loads(line))
    return claims


def _outcome_bucket(claim: dict) -> str:
    """
    Collapses verdict + debate_resolution into one of 4 outcome
    buckets for stratification: GROUNDED (never challenged), UPHELD
    (challenged, debate upheld), REJECTED (challenged, debate
    rejected), BLOCKED (no evidence found at all).
    """
    verdict = (claim.get("verdict") or "").lower()
    resolution = claim.get("debate_resolution")
    if verdict == "grounded":
        return "GROUNDED"
    if verdict == "blocked":
        return "BLOCKED"
    if verdict == "challenged":
        if resolution == "upheld":
            return "UPHELD"
        if resolution == "rejected":
            return "REJECTED"
        return "CHALLENGED_UNRESOLVED"  # should not normally occur in a completed run
    return "UNKNOWN"


def ensure_unique_claim_ids(claims: list[dict]) -> list[dict]:
    """
    claim_id is built from ticker + period + the first 40 characters of the
    claim text, so two similar claims in one scenario could share an id.
    Duplicate ids would silently overwrite each other in the answer key and
    produce two worksheet rows that join to the same verdict. Later
    duplicates get a numeric suffix (__2, __3, ...); the first keeps its id.
    """
    seen: dict[str, int] = {}
    for c in claims:
        base = c["claim_id"]
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            c["claim_id"] = f"{base}__{seen[base]}"
    return claims


def stratify(claims: list[dict]) -> dict[tuple, list[dict]]:
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for c in claims:
        key = (c["ticker"], c["claim_category"], _outcome_bucket(c))
        strata[key].append(c)
    return strata


def stratified_sample(claims: list[dict], target_n: int, seed: int = _RANDOM_SEED) -> list[dict]:
    rng = random.Random(seed)
    strata = stratify(claims)

    # Pass 1: guarantee at least 1 from every non-empty stratum, so rare
    # combinations are still represented in the final sample.
    sampled: list[dict] = []
    remaining_pool: dict[tuple, list[dict]] = {}
    for key, members in strata.items():
        shuffled = members[:]
        rng.shuffle(shuffled)
        take = shuffled[:1]
        sampled.extend(take)
        remaining_pool[key] = shuffled[1:]

    # Pass 2: fill remaining budget proportionally to each stratum's
    # remaining size, until target_n is reached or the pool is exhausted.
    budget = target_n - len(sampled)
    total_remaining = sum(len(v) for v in remaining_pool.values())

    if budget > 0 and total_remaining > 0:
        # Proportional allocation with largest-remainder rounding, so the
        # allocated counts sum exactly to `budget` (not over/under due to
        # independent per-stratum rounding).
        raw_allocations = {
            key: (len(members) / total_remaining) * budget for key, members in remaining_pool.items()
        }
        allocations = {key: int(v) for key, v in raw_allocations.items()}
        shortfall = budget - sum(allocations.values())
        remainders = sorted(raw_allocations.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True)
        for key, _ in remainders[:shortfall]:
            allocations[key] += 1

        for key, members in remaining_pool.items():
            n = min(allocations.get(key, 0), len(members))
            sampled.extend(members[:n])

    return sampled


def write_worksheet(sample: list[dict], path: Path) -> None:
    fieldnames = [
        "claim_id",
        "ticker",
        "time_period",
        "claim_category",
        "claim_text",
        "retrieved_evidence",
        "human_label",
        "notes",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in sample:
            writer.writerow(
                {
                    "claim_id": c["claim_id"],
                    "ticker": c["ticker"],
                    "time_period": c["time_period"],
                    "claim_category": c["claim_category"],
                    "claim_text": c["claim_text"],
                    "retrieved_evidence": c.get("retrieved_evidence") or "",
                    "human_label": "",  # fill in with one of: SUPPORTED / CONTRADICTED / UNSUPPORTED / UNVERIFIABLE
                    "notes": "",
                }
            )


def write_answer_key(sample: list[dict], path: Path) -> None:
    answer_key = {
        c["claim_id"]: {
            "verdict": c.get("verdict"),
            "debate_resolution": c.get("debate_resolution"),
            "grounding_score": c.get("grounding_score"),
            "is_usable": c.get("is_usable"),
            "outcome_bucket": _outcome_bucket(c),
        }
        for c in sample
    }
    with open(path, "w") as f:
        json.dump(answer_key, f, indent=2)


def print_stratum_summary(sample: list[dict]) -> None:
    print("\nSample composition by stratum:")
    counts: dict[tuple, int] = defaultdict(int)
    for c in sample:
        counts[(c["ticker"], c["claim_category"], _outcome_bucket(c))] += 1
    for key in sorted(counts.keys()):
        ticker, category, outcome = key
        print(f"  {ticker:<6} {category:<10} {outcome:<12} n={counts[key]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample claims for human spot-check evaluation.")
    parser.add_argument("claims_path", nargs="?", default=None, help="Path to claims_*.jsonl")
    parser.add_argument("--n", type=int, default=_DEFAULT_SAMPLE_SIZE, help="Target sample size")
    args = parser.parse_args()

    claims_path = Path(args.claims_path) if args.claims_path else find_latest_claims_file()
    print(f"Loading claims from: {claims_path}")
    claims = ensure_unique_claim_ids(load_claims(claims_path))
    print(f"Loaded {len(claims)} total VERIFY-FIN claims.")

    if len(claims) < args.n:
        print(
            f"WARNING: only {len(claims)} claims available, fewer than the requested sample "
            f"size of {args.n}. The full claim population will be used as the sample instead."
        )

    sample = stratified_sample(claims, target_n=args.n)
    # Shuffle row order (fixed seed) so labelling only the first part of the worksheet
    # still gives a random subset rather than whole strata in a row.
    random.Random(_RANDOM_SEED).shuffle(sample)
    print(f"\nDrew a stratified sample of {len(sample)} claims (target was {args.n}).")
    print_stratum_summary(sample)

    output_dir = resolve_path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    worksheet_path = output_dir / "human_eval_worksheet.csv"
    answer_key_path = output_dir / "human_eval_answer_key.json"

    write_worksheet(sample, worksheet_path)
    write_answer_key(sample, answer_key_path)

    print(f"\nBLIND labeling worksheet written to: {worksheet_path}")
    print(f"  -> Open this file and fill in the 'human_label' column for every row.")
    print(f"  -> Allowed values: {_HUMAN_LABEL_OPTIONS}")
    print(f"\nHidden answer key written to: {answer_key_path}")
    print(f"  -> Do NOT open this until labelling is complete -- it contains VERIFY-FIN's own verdicts.")
    print(f"\nWhen labelling is done, run: python scripts/score_human_eval.py")


if __name__ == "__main__":
    main()
