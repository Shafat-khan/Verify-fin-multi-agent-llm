# VERIFY-FIN: Claim-Level Grounding for Multi-Agent Financial Analysis

> An MSc Data Science research project investigating whether claim-level evidence verification can improve the auditability and reliability of LLM-generated financial analysis.

## Overview

Large language models can generate fluent financial analysis containing statements that are unsupported, incorrect, or difficult to audit. VERIFY-FIN explores a different approach: instead of checking only a completed report, it represents factual statements as structured claims and verifies them against traceable financial evidence **before those claims reach the final synthesis stage**.

The system is implemented as a LangGraph `StateGraph` with specialised financial-analysis agents, a retrieval-based Verification Agent, a Debate Agent for borderline claims, and a Synthesis Agent that receives only claims allowed through the evidence gate.

The project uses public financial and macroeconomic sources, including SEC EDGAR/XBRL, the Federal Reserve FRED API, and financial-news headlines. Evidence is indexed in scenario-specific ChromaDB collections to reduce the risk of mixing information across companies or historical quarters.

## Research Question

**Does claim-level verification reduce the rate of unsupported or incorrect financial claims compared with conventional single-agent and unverified multi-agent LLM systems?**

The project compares three conditions:

1. **Single-Agent baseline** — a no-tool, free-form LLM generation condition.
2. **Unverified Multi-Agent baseline** — the same data-grounded analyst pipeline without the verification and debate stages.
3. **VERIFY-FIN** — the multi-agent pipeline with claim-level verification, debate for borderline claims, and evidence-gated synthesis.

## System Architecture

The current pipeline is:

```text
Orchestrator
    |
    v
Fundamental Agent
    |
    v
Macro Agent
    |
    v
Sentiment Agent
    |
    v
Verification Agent
    |
    +---- GROUNDED -------------------------+
    |                                      |
    +---- CHALLENGED --> Debate Agent ------+--> Synthesis Agent
    |                 upheld claims        |    (usable claims only)
    |                                      |
    +---- BLOCKED --------------------------X
```

The analyst agents are conceptually specialised, but the current implementation connects them as sequential graph nodes. This design was chosen to keep the research prototype manageable and to accommodate API rate limits.

### Agent responsibilities

- **Orchestrator:** Routes tasks and coordinates the pipeline.
- **Fundamental Agent:** Produces company financial claims using SEC EDGAR/XBRL data.
- **Macro Agent:** Produces macroeconomic claims using FRED observations.
- **Sentiment Agent:** Analyses financial-news headlines.
- **Verification Agent:** Retrieves evidence and scores claim grounding.
- **Debate Agent:** Reviews borderline claims and returns an uphold/reject decision.
- **Synthesis Agent:** Produces the final report using only claims accepted by the evidence gate.

## Claim-Level Verification

Analyst outputs are represented as structured `ClaimObject` records rather than unrestricted prose. The schema includes:

- `claim_text`
- `claim_category`
- `produced_by`
- `source_reference`
- `confidence_score`
- `verdict`
- `debate_resolution`

A source reference is mandatory for claims produced by the structured analyst pipeline. Claims are routed according to their grounding score:

| Grounding score | Verdict | Handling |
|---|---|---|
| `>= 0.75` | `GROUNDED` | Eligible for synthesis |
| `0.55 to < 0.75` | `CHALLENGED` | Sent to the Debate Agent |
| `< 0.55` | `BLOCKED` | Excluded from synthesis |

The verifier combines semantic retrieval with additional checks for numerical claims. It uses sentence-transformer embeddings and retrieves candidate evidence from ChromaDB. Numerical matching, value variants, and date-qualified evidence are used because semantic similarity alone may retrieve text that sounds relevant without supporting the exact value.

Only `GROUNDED` claims and `CHALLENGED` claims subsequently upheld by debate are eligible for final synthesis.

## Data Sources and Retrieval

| Source | Role in the system |
|---|---|
| SEC EDGAR / XBRL Company Facts | Company financial-statement figures |
| SEC 10-K and 10-Q filings | Filing text and tables used as verification evidence |
| Federal Reserve FRED API | Macroeconomic time series |
| Google News RSS / financial-news headlines | Sentiment analysis and supporting evidence |

The project uses a scenario-specific ChromaDB collection for each company-quarter scenario. This helps keep evidence aligned with the historical period being evaluated and reduces cross-scenario contamination.

The implementation uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings and ChromaDB for semantic retrieval.

## Technology Stack

- **Language:** Python
- **Agent orchestration:** LangGraph
- **LLM inference:** Groq API; the interim report records GPT-OSS-120B as the model used in the current evaluation
- **Structured outputs:** Pydantic
- **Retrieval-augmented generation:** ChromaDB
- **Embeddings:** Sentence Transformers
- **Financial data:** SEC EDGAR/XBRL
- **Macroeconomic data:** FRED API
- **News:** Google News RSS / financial-news headlines
- **Development environment:** Google Colab

Model availability and API limits can change. Configure credentials and model identifiers for the environment in which you run the project.

## Interim Evaluation

The interim report describes a controlled preliminary evaluation across **14 valid historical-quarter scenarios** covering five companies. The original target was 15 scenarios, but only two valid quarters were resolved for XOM under the current data-selection rules.

### Preliminary reported results

| Condition | Post-hoc unsupported-claim rate | Grounded/usable rate | Mean runtime | Evaluable scenarios |
|---|---:|---:|---:|---:|
| Single-Agent | 99.38% ± 1.64%* | 0.7% | 0.70 s | 7/14 |
| Unverified Multi-Agent | 8.00% ± 6.01% | 77.8% | 11.38 s | 14/14 |
| VERIFY-FIN | 10.20% (95% CI: 7.59–12.81%) | 79.5% | 12.78 s | 14/14 |

\*The Single-Agent figure is based on only seven evaluable scenarios and is an automated post-hoc evidence-support measurement, not an independently established hallucination rate.

### How to interpret these results

The current evaluation **does not demonstrate a statistically significant reduction** in the aggregate post-hoc unsupported-claim rate for VERIFY-FIN compared with the Unverified Multi-Agent baseline. The interim paired comparison reports a one-sided Wilcoxon p-value of 0.856.

The principal interim contribution is the implementation of an auditable, claim-level evidence gate that operates before synthesis. Because the unverified multi-agent baseline already uses deterministic data sourcing for numerical and macroeconomic claims, its raw claim quality can be similar to VERIFY-FIN's. The verification layer's value must therefore also be evaluated by how accurately it identifies unsupported claims and prevents them from entering the final report.

**Important terminology:** a claim that cannot be supported by the available knowledge base is not automatically false. The interim report therefore uses *post-hoc unsupported-claim rate* rather than presenting these automated labels as definitive factuality ground truth.

## Engineering and Testing

The interim report records **220 automated tests passing** at the reported development stage. It also documents engineering fixes involving:

- HTML scrubbing and filing parsing
- XBRL tag aliases and historical-period selection
- Row-level table chunking and preservation of table headers
- Quarter-versus-year-to-date duration handling
- Date-qualified macroeconomic evidence matching
- Separate cache namespaces for evaluation conditions
- LLM extraction truncation and API retry/backoff handling

A known limitation remains in numeric substring matching, where a value such as `4.3` may match inside `4.33`.

## Reproducibility

The reports describe a Google Colab-based implementation. The source-code files and exact project configuration should be included in the repository before presenting it as independently reproducible.

A typical setup will require:

1. Python 3.10 or later.
2. The dependencies listed in `requirements.txt`.
3. API credentials configured as environment variables or Colab secrets.
4. Access to SEC EDGAR, FRED, and the selected financial-news source.
5. The project's scenario configuration, prompts, and evaluation scripts.
6. Explicitly pinned historical filings, quarter definitions, and evidence cut-off dates.

Never commit API keys, `.env` files, personal credentials, or private evaluation data. Exact package versions and configuration should be pinned when preparing a reproducible release.

## Limitations and Future Work

The interim report identifies the following priorities for the final evaluation:

- Complete Single-Agent claim extraction and evaluation for all resolvable scenarios.
- Enforce evidence-date cut-offs to prevent temporal leakage.
- Independently annotate a sample of claims as `SUPPORTED`, `CONTRADICTED`, `UNSUPPORTED`, or `UNVERIFIABLE`.
- Compare automated verdicts with human labels using precision, recall, F1, confusion matrices, and Cohen's kappa.
- Conduct an ablation study comparing semantic-only retrieval, hybrid verification, and hybrid verification with debate.
- Analyse errors by claim category, especially sentiment and historical macroeconomic claims.
- Investigate residual numerical matching edge cases.
- Report results with appropriate confidence intervals and statistical tests.

These are planned or outstanding research tasks, not completed results.

## Responsible Use

VERIFY-FIN is a research prototype for studying evidence-grounded financial analysis. Its outputs are **not financial advice** and should not be used as the sole basis for investment decisions. Evidence retrieval and automated verification can miss relevant information, accept incorrect claims, or reject valid claims when the knowledge base is incomplete.

## Project Documentation

The repository can include the following supporting documents:

- `README.md` — project overview and technical documentation.
- `VERIFY_FIN_Interim_Report.pdf` — interim implementation and preliminary evaluation report.
- `PgProposalShafatJan26.pdf` — original MSc project proposal.
- Source modules and notebooks — agent definitions, retrieval and verification logic, data pipeline, and evaluation harness.

Before making the repository public, remove student identifiers or other personal information from the PDFs.

## Skills Demonstrated

Multi-Agent Systems · Large Language Models · Retrieval-Augmented Generation (RAG) · Claim Verification · LangGraph · Pydantic · ChromaDB · Sentence Transformers · Python · Financial Data Engineering · API Integration · Evaluation Design · Statistical Analysis · Responsible AI

## Academic Context

Developed as an MSc Data Science research project at the University of Greenwich. The project investigates claim-level grounding as an architectural mechanism for improving traceability and evaluating factual support in multi-agent LLM financial analysis.
