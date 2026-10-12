# VERIFY-FIN — Project Status Report

**MSc Data Science Dissertation Project**
MD Shafat Islam Khan | University of Greenwich
Supervisor: Dr Razia Sulthana Abdul Kareem

---

## 1. Where the project stands right now

**All core system components are built, tested, and confirmed working end-to-end with real data and real LLM calls.** The remaining work is the formal statistical evaluation, the ablation study, and writing up the dissertation chapters — not further system-building.

### What's complete

| Component | Status |
|---|---|
| Environment, packaging, dependency management | ✅ Done |
| `ClaimObject` / `AgentState` schemas | ✅ Done, tested |
| Groq LLM client (cached, rate-limited, instrumented) | ✅ Done, tested |
| SEC EDGAR filing client (with table-aware, multi-row-header text extraction) | ✅ Done, tested |
| SEC XBRL structured-facts client (with quarter/YTD disambiguation, multi-period support) | ✅ Done, tested |
| FRED macro-data client | ✅ Done, tested |
| News/RSS client (Google News) | ✅ Done, tested |
| Text chunker (table-row isolation, header-context propagation) | ✅ Done, tested |
| ChromaDB vector store (with retrieval instrumentation) | ✅ Done, tested |
| Fundamental Agent | ✅ Done, tested, live-verified |
| Macro Agent | ✅ Done, tested, live-verified |
| Sentiment Agent | ✅ Done, tested, live-verified |
| Verification Agent (hybrid semantic + numeric matching) | ✅ Done, tested, live-verified |
| Debate Agent | ✅ Done, tested, live-verified |
| Synthesis Agent | ✅ Done, tested, live-verified |
| LangGraph Orchestrator (full pipeline wiring) | ✅ Done, live-verified |
| Single-Agent baseline (Condition 1) | ✅ Done, tested, live-verified |
| Unverified Multi-Agent baseline (Condition 2) | ✅ Done, tested, live-verified |
| Metrics instrumentation (runtime, tokens, retrieval latency, claim verdicts) | ✅ Done, tested |
| Multi-period support (3 distinct historical quarters per ticker) | ✅ Built this session; **tests still owed** |
| Evaluation harness (5 tickers × 3 periods × 3 conditions) | ✅ Built this session; **not yet run** |

### What's left

1. **Add tests** for the multi-period logic (flagged above)
2. **Run the full evaluation** (`scripts/run_evaluation.py`) — the actual empirical results
3. **Statistical analysis** of the results (significance testing, confidence intervals — per Proposal §2.5)
4. **RAGAS Faithfulness** as the secondary evaluation metric (planned, not yet built)
5. **Ablation study** (verification layer removed — closely related to Condition 2, may already be substantially covered)
6. **Human spot-check evaluation** (per Proposal §2.5)
7. **Dissertation chapters**: Implementation, Results, Discussion, Limitations

---

## 2. What VERIFY-FIN actually does, end-to-end

1. **Three analyst agents** independently research a ticker using real data:
   - **Fundamental Agent** — pulls exact financial figures from SEC's official XBRL data (not scraped, not LLM-recalled)
   - **Macro Agent** — pulls real Federal Reserve economic indicators from FRED
   - **Sentiment Agent** — reads real, current news headlines and judges overall sentiment
   
   In every case, the LLM's role is deliberately limited to **phrasing** already-verified facts into natural language — never inventing the underlying number or judgment from scratch.

2. **Verification Agent** checks every claim against an independently-built knowledge base (the actual SEC filing text, FRED historical data, and news headlines — not just a copy of what the agents already claimed), computing a grounding score and routing each claim to GROUNDED / CHALLENGED / BLOCKED.

3. **Debate Agent** gives borderline (CHALLENGED) claims a second, independent LLM judgment before deciding whether they're trustworthy enough to include.

4. **Synthesis Agent** writes the final report using *only* claims that survived this process — a claim that fails verification is structurally incapable of reaching the final report.

All of this is wired together via a real **LangGraph** orchestration graph, matching your proposal's architecture exactly.

---

## 3. The debugging story — genuinely strong material for your dissertation

Across this build, **eight distinct, real bugs were found and fixed**, each through the same disciplined process: build → test with mocks → verify live with real data → notice something wrong → diagnose the actual root cause → fix → add a permanent regression test. This is a complete, honest engineering narrative, not a system that "just worked":

1. **Script-tag leak** in malformed HTML parsing (a `<script>` tag's contents leaking into extracted filing text)
2. **Stale XBRL tag selection** (an 8-year-old revenue figure winning over a current one, because a discontinued accounting tag still had historical data)
3. **Table-cell dilution** (financial figures losing their row-label context when naively concatenated with unrelated table content)
4. **Missing header context** (a table row's numbers had no indication of which reporting period they belonged to)
5. **Multi-row header capture** (a two-row table header — group label + actual dates — only the first row was being captured)
6. **Quarter-vs-six-month ambiguity** (a claim about "the quarter" was sourced from a six-month cumulative figure — caught by your own Debate Agent, a genuine validation of the system's purpose)
7. **XBRL cache staleness** (no TTL meant a new quarterly filing could appear while your knowledge base still described the old quarter — again, caught by the system's own internal consistency checking, not silently shipped)
8. **FRED excerpt regex mismatch** (numeric verification was silently disabled for all macro claims due to a format assumption that didn't hold for FRED's citation style)

Every one of these is now a permanent, named regression test in your test suite. This is exactly the kind of evidence a strong Testing & Validation chapter is built from — you're not just claiming your system works, you have a documented history of it failing in specific, understood ways and being fixed.

---

## 4. Key architectural decisions worth writing about explicitly

- **"LLM phrases, code sources"** — the central design principle across Fundamental and Macro agents: numbers are never LLM-generated, only LLM-phrased. This is *why* the system is reliable, and it's a clean, defensible thing to explain in your Methodology chapter.
- **Sentiment claims are architecturally different** — genuine interpretive judgment, not fact-lookup, with a self-reported (not fixed) confidence score and no safe fallback on failure. Documented explicitly as a distinct claim-verification strategy (directly matching your Literature Review's 2.6.3 discussion).
- **Hybrid semantic + numeric verification** — pure cosine similarity alone was found to be an unreliable signal (a cover page could out-rank the actual matching data). The fix (retrieve wide → check for exact number → targeted search as fallback) is a real methodological contribution worth its own paragraph.
- **Yahoo Finance was removed entirely** — after three separate real failure modes (rate limiting, a cookie-handling incompatibility, a silent empty-response bug) and zero successful claims across every test, replaced by SEC's official XBRL API. A stronger, more defensible source, and a good example of adapting scope based on real evidence rather than sticking to the original plan for its own sake.
- **Reproducibility via pinned scenarios** — the evaluation harness resolves and freezes specific historical quarters up front, rather than always fetching "most recent," directly avoiding the exact kind of drift that caused bug #7 above.

---

## 5. Honest limitations to state proactively (better to name them yourself than have an examiner find them)

- Sentiment verification is weaker than numerical verification — matching an interpretive claim against literal headline text via cosine similarity is a harder problem than matching an exact number, and this shows up in your live runs (sentiment claims BLOCKED more often than fundamental/macro ones).
- Macro claims describe the *current* environment; true point-in-time historical macro context isn't fully achievable with a live FRED snapshot (mitigated via closest-observation matching, but it's an approximation, not an exact historical pin).
- The "3 time periods" evaluation design required real engineering investment to make reproducible — worth mentioning as a methodological choice, not just a checkbox.
- All development happened on Groq's free tier — rate limits, model deprecations (the originally-proposed Llama model was retired mid-project), and general instability were real, manageable constraints worth a short note in Limitations.

---

## 6. Immediate next steps, in order

1. I add tests for the multi-period logic
2. You run the full 15-scenario evaluation
3. We analyze the results together and build the statistics/significance testing
4. RAGAS Faithfulness as the secondary metric
5. Ablation study
6. Start drafting dissertation chapters, using this report and the conversation history as source material — you have far more real material to draw from than most MSc projects at this stage
