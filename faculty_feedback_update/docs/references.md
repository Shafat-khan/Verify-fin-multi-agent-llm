# References and citation plan (Harvard style, matching your Chapter 2)

> **Please verify before submission.** I have no internet access in my working environment, so every
> entry below is written from memory and standard bibliographic knowledge, not checked against the
> publisher's page. I am confident in the authors, titles and venues of most entries; entries marked
> **[verify]** are ones where I am less sure of a detail (page range, arXiv number, venue or year).
> Check each one against the original paper's own "cite as" information. **[NEW]** = not in your
> current Chapter 2 reference list.

---

## Part A. Where to cite what

| Chapter / section | Claim you are making | Cite |
|---|---|---|
| 1 Introduction | LLMs produce fluent unsupported text ("hallucination") | Ji et al. (2023); Huang et al. (2023); Zhang et al. (2023) |
| 1 Introduction | Hallucination is a documented problem specifically in finance | Kang and Liu (2023) [verify]; Levy et al. (2025) |
| 1 Introduction | Errors propagate between agents in multi-agent pipelines | Joshi (2025); Wu et al. (2023) |
| 2 Literature review | Multi-agent financial systems | Shao et al. / Xiao et al. (2024/2025) **see Part C**; Yu et al. (2024) **[NEW]**; Fatouros et al. (2024) **[NEW]**; Chen et al. (2025); FinDebate (2025) |
| 2 Literature review | Finance-domain models and benchmarks | Wu et al. (2023) BloombergGPT **[NEW]**; Chen et al. (2021) FinQA **[NEW]**; Islam et al. (2023) FinanceBench **[NEW]**; Liu et al. (2020) |
| 2 Literature review | Attribution to sources as the right notion of "supported" | Rashkin et al. (2023) **[NEW]** |
| 2 Literature review | Verification and self-checking methods | Dhuliawala et al. (2023) **[NEW]**; Manakul et al. (2023); Asai et al. (2023); Min et al. (2023) |
| 2 Literature review | Debate / multi-agent adjudication | Du et al. (2024) **[NEW]**; Zheng et al. (2023) **[NEW]** (LLM-as-judge) |
| 3 Methodology: knowledge base | RAG; sentence embeddings; MiniLM | Lewis et al. (2020); Gao et al. (2023); Reimers and Gurevych (2019); Wang et al. (2020) **[NEW]** |
| 3 Methodology: data | SEC XBRL company facts; FRED | U.S. SEC (2026) **[NEW]**; Federal Reserve Bank of St. Louis (2026) **[NEW]** |
| 3 Methodology: implementation | LangGraph; ChromaDB; the generator model | LangChain (2024) **[NEW]**; Chroma (2024) **[NEW]**; OpenAI (2025) **[NEW]** [verify] |
| 3 Methodology: claim extraction (baselines) | Decomposing text into atomic claims | Min et al. (2023) |
| 3 Methodology: claim verification | Claim-level verification datasets and task framing | Thorne et al. (2018) |
| 3 Methodology: **fault injection** | Controlled corruption with ground truth by construction | DeMillo et al. (1978) **[NEW]**; Jia and Harman (2011) **[NEW]**; Ribeiro et al. (2020) **[NEW]**; Thorne et al. (2018) (refuted claims built by mutating true ones) |
| 3 Methodology: human evaluation | Inter-annotator / human-system agreement; the kappa scale | Cohen (1960) **[NEW]**; Landis and Koch (1977) **[NEW]**; Artstein and Poesio (2008) **[NEW]** |
| 3 Methodology: statistics | Wilson interval | Wilson (1927) **[NEW]** |
| 3 Methodology: statistics | Wilcoxon signed-rank; zeros and ties; exact permutation | Wilcoxon (1945) **[NEW]**; Pratt (1959) **[NEW]** |
| 3 Methodology: statistics | Rank-biserial effect size | Kerby (2014) **[NEW]** |
| 3 Methodology: statistics | McNemar's test | McNemar (1947) **[NEW]** |
| 3 Methodology: statistics | Bootstrap; cluster bootstrap | Efron (1979) **[NEW]**; Efron and Tibshirani (1993) **[NEW]**; Field and Welsh (2007) **[NEW]** |
| 3 Methodology: software | SciPy, scikit-learn, NumPy | Virtanen et al. (2020); Pedregosa et al. (2011); Harris et al. (2020) **[NEW]** |
| 4/5 Results, Discussion | Why "supported by the knowledge base" is not "true" | Rashkin et al. (2023); Thorne et al. (2018) |
| 6 Limitations / future work | RAGAS-style faithfulness as a complementary measure | Es et al. (2023) |

---

## Part B. Reference list

**Alphabetical, Harvard style. Entries without a mark are already in your Chapter 2 and are reproduced unchanged.**

Artstein, R. and Poesio, M. (2008) Inter-coder agreement for computational linguistics. *Computational Linguistics*, 34(4), pp. 555–596. **[NEW]**

Asai, A., Wu, Z., Wang, Y., Sil, A. and Hajishirzi, H. (2023) Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. arXiv preprint arXiv:2310.11511. Available at: https://arxiv.org/abs/2310.11511 (Accessed: May 2026).

Chen, Z., et al. (2025) AlphaAgents: Large Language Model based Multi-Agents for Equity Portfolio Constructions. arXiv preprint arXiv:2508.11152. Available at: https://arxiv.org/html/2508.11152v1 (Accessed: May 2026).

Chen, Z., Chen, W., Smiley, C., Shah, S., Borova, I., Langdon, D., Moussa, R., Beane, M., Huang, T.-H., Routledge, B. and Wang, W.Y. (2021) FinQA: A dataset of numerical reasoning over financial data. *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 3697–3711. **[NEW]**

Chroma (2024) *Chroma: the open-source embedding database*. Available at: https://www.trychroma.com/ (Accessed: October 2026). **[NEW]**

Cohen, J. (1960) A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), pp. 37–46. **[NEW]**

DeMillo, R.A., Lipton, R.J. and Sayward, F.G. (1978) Hints on test data selection: help for the practicing programmer. *Computer*, 11(4), pp. 34–41. **[NEW]**

Dhuliawala, S., Komeili, M., Xu, J., Raileanu, R., Li, X., Celikyilmaz, A. and Weston, J. (2023) Chain-of-Verification Reduces Hallucination in Large Language Models. arXiv preprint arXiv:2309.11495. Available at: https://arxiv.org/abs/2309.11495 (Accessed: October 2026). **[NEW]**

Du, Y., Li, S., Torralba, A., Tenenbaum, J.B. and Mordatch, I. (2024) Improving factuality and reasoning in language models through multiagent debate. *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235. arXiv:2305.14325. **[NEW]** [verify]

Efron, B. (1979) Bootstrap methods: another look at the jackknife. *The Annals of Statistics*, 7(1), pp. 1–26. **[NEW]**

Efron, B. and Tibshirani, R.J. (1993) *An Introduction to the Bootstrap*. New York: Chapman and Hall. **[NEW]**

Es, S., James, J., Espinosa-Anke, L. and Schockaert, S. (2023) RAGAS: Automated Evaluation of Retrieval Augmented Generation. arXiv preprint arXiv:2309.15217. Available at: https://arxiv.org/abs/2309.15217 (Accessed: May 2026).

Fatouros, G., Metaxas, K., Soldatos, J. and Kyriazis, D. (2024) Can Large Language Models Beat Wall Street? Unveiling the Potential of AI in Stock Selection. arXiv preprint arXiv:2401.03737. Available at: https://arxiv.org/abs/2401.03737 (Accessed: October 2026). **[NEW]** [verify]

Federal Reserve Bank of St. Louis (2026) *FRED Economic Data* (series FEDFUNDS, CPIAUCSL, UNRATE, GDP). Available at: https://fred.stlouisfed.org/ (Accessed: October 2026). **[NEW]**

Field, C.A. and Welsh, A.H. (2007) Bootstrapping clustered data. *Journal of the Royal Statistical Society: Series B*, 69(3), pp. 369–390. **[NEW]**

FinDebate Research Group (2025) FinDebate: Multi-Agent Collaborative Intelligence for Financial Analysis. arXiv preprint arXiv:2509.17395. Available at: https://arxiv.org/pdf/2509.17395 (Accessed: May 2026).

Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J. and Wang, H. (2023) Retrieval-Augmented Generation for Large Language Models: A Survey. arXiv preprint arXiv:2312.10997. Available at: https://arxiv.org/abs/2312.10997 (Accessed: May 2026).

Harris, C.R., Millman, K.J., van der Walt, S.J., et al. (2020) Array programming with NumPy. *Nature*, 585, pp. 357–362. **[NEW]**

Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang, H., Chen, Q., Peng, W., Feng, X., Qin, B. and Liu, T. (2023) A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions. arXiv preprint arXiv:2311.05232. Available at: https://arxiv.org/abs/2311.05232 (Accessed: May 2026).

Islam, P., Kannappan, A., Kiela, D., Qian, R., Scherrer, N. and Vidgen, B. (2023) FinanceBench: A New Benchmark for Financial Question Answering. arXiv preprint arXiv:2311.11944. Available at: https://arxiv.org/abs/2311.11944 (Accessed: October 2026). **[NEW]**

Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y.J., Madotto, A. and Fung, P. (2023) Survey of hallucination in natural language generation. *ACM Computing Surveys*, 55(12), Article 248, pp. 1–38. **[NEW]**

Jia, Y. and Harman, M. (2011) An analysis and survey of the development of mutation testing. *IEEE Transactions on Software Engineering*, 37(5), pp. 649–678. **[NEW]**

Joshi, S. (2025) Architectures and Challenges of AI Multi-Agent Frameworks for Financial Services. *Current Journal of Applied Science and Technology*, 44(6), pp. 52–72. Available at: https://journalcjast.com/index.php/CJAST/article/view/4558 (Accessed: May 2026).

Kang, H. and Liu, X.-Y. (2023) Deficiency of Large Language Models in Finance: An Empirical Examination of Hallucination. arXiv preprint arXiv:2311.15548. Available at: https://arxiv.org/abs/2311.15548 (Accessed: October 2026). **[NEW]** [verify]

Kerby, D.S. (2014) The simple difference formula: an approach to teaching nonparametric correlation. *Comprehensive Psychology*, 3, Article 11.IT.3.1. **[NEW]**

Kumar, S.N.P., et al. (2025) Multi-Agent AI Systems in Finance: Models, Applications, and Challenges. *International Journal of Advanced Research in Computer Science and Technology*. Available at: https://ijarcst.org/index.php/ijarcst/article/view/246 (Accessed: May 2026).

LangChain (2024) *LangGraph documentation*. Available at: https://langchain-ai.github.io/langgraph/ (Accessed: October 2026). **[NEW]**

Landis, J.R. and Koch, G.G. (1977) The measurement of observer agreement for categorical data. *Biometrics*, 33(1), pp. 159–174. **[NEW]**

Levy, M., et al. (2025) Large Language Model Agents for Investment Management: Foundations, Benchmarks, and Research Frontiers. *Proceedings of ICAIF 2025*. ACM Digital Library. Available at: https://dl.acm.org/doi/10.1145/3768292.3770387 (Accessed: May 2026).

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S. and Kiela, D. (2020) Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems*, 33, pp. 9459–9474. Available at: https://arxiv.org/abs/2005.11401 (Accessed: May 2026).

Liu, Z., Huang, D., Huang, K., Li, Z. and Zhao, J. (2020) FinBERT: A Pre-trained Financial Language Representation Model for Financial Text Mining. *Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence (IJCAI-20)*, pp. 4513–4519. Available at: https://arxiv.org/abs/1908.10063 (Accessed: May 2026).

Manakul, P., Liusie, A. and Gales, M.J.F. (2023) SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. *Proceedings of EMNLP 2023*, pp. 9004–9017. Available at: https://arxiv.org/abs/2303.08896 (Accessed: May 2026).

McNemar, Q. (1947) Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), pp. 153–157. **[NEW]**

Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W., Koh, P.W., Iyyer, M., Zettlemoyer, L. and Hajishirzi, H. (2023) FActScore: Fine-Grained Atomic Evaluation of Factual Precision in Long Form Text Generation. *Proceedings of EMNLP 2023*, pp. 12076–12100. Available at: https://arxiv.org/abs/2305.14251 (Accessed: May 2026).

Nguyen, P. and Pham, T. (2025) Toward Reliable Evaluation of LLM-Based Financial Multi-Agent Systems: Taxonomy, Coordination Primacy, and Cost Awareness. arXiv preprint arXiv:2603.27539. Available at: https://arxiv.org/pdf/2603.27539 (Accessed: May 2026). **(see Part C: the arXiv number suggests 2026, and the in-text citations are inconsistent)**

OpenAI (2025) *gpt-oss-120b and gpt-oss-20b model card*. arXiv preprint arXiv:2508.10925. Available at: https://arxiv.org/abs/2508.10925 (Accessed: October 2026). **[NEW]** [verify]

Park, J.S., O'Brien, J.C., Cai, C.J., Morris, M.R., Liang, P. and Bernstein, M.S. (2023) Generative Agents: Interactive Simulacra of Human Behavior. *Proceedings of UIST 2023*. Available at: https://arxiv.org/abs/2304.03442 (Accessed: May 2026).

Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011) Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, 12, pp. 2825–2830. **[NEW]**

Pratt, J.W. (1959) Remarks on zeros and ties in the Wilcoxon signed rank procedures. *Journal of the American Statistical Association*, 54(287), pp. 655–667. **[NEW]**

Rashkin, H., Nikolaev, V., Lamm, M., Aroyo, L., Collins, M., Das, D., Petrov, S., Tomar, G.S., Turc, I. and Reitter, D. (2023) Measuring attribution in natural language generation models. *Computational Linguistics*, 49(4), pp. 777–840. **[NEW]** [verify pages]

Reimers, N. and Gurevych, I. (2019) Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *Proceedings of EMNLP 2019*, pp. 3982–3992. Available at: https://arxiv.org/abs/1908.10084 (Accessed: May 2026).

Ribeiro, M.T., Wu, T., Guestrin, C. and Singh, S. (2020) Beyond accuracy: behavioral testing of NLP models with CheckList. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)*, pp. 4902–4912. **[NEW]**

Shao, Y., Wang, Y., Sun, R., et al. (2025) TradingAgents: Multi-Agent LLM Financial Trading Framework. *Proceedings of AAAI 2025*. Available at: https://arxiv.org/pdf/2412.20138 (Accessed: May 2026). **(see Part C: check the author list and venue)**

Thorne, J., Vlachos, A., Christodoulopoulos, C. and Mittal, A. (2018) FEVER: a Large-Scale Dataset for Fact Extraction and VERification. *Proceedings of NAACL-HLT 2018*, pp. 809–819. Available at: https://arxiv.org/abs/1803.05355 (Accessed: May 2026).

U.S. Securities and Exchange Commission (2026) *EDGAR application programming interfaces (XBRL company facts)*. Available at: https://www.sec.gov/search-filings/edgar-application-programming-interfaces (Accessed: October 2026). **[NEW]** [verify URL]

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L. and Polosukhin, I. (2017) Attention Is All You Need. *Advances in Neural Information Processing Systems*, 30, pp. 5998–6008. Available at: https://arxiv.org/abs/1706.03762 (Accessed: May 2026).

Virtanen, P., Gommers, R., Oliphant, T.E., et al. (2020) SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods*, 17, pp. 261–272. **[NEW]**

Wang, W., Wei, F., Dong, L., Bao, H., Yang, N. and Zhou, M. (2020) MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. *Advances in Neural Information Processing Systems*, 33, pp. 5776–5788. **[NEW]** [verify pages]

Wilcoxon, F. (1945) Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), pp. 80–83. **[NEW]**

Wilson, E.B. (1927) Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association*, 22(158), pp. 209–212. **[NEW]**

Wu, Q., Bansal, G., Zhang, J., Wu, Y., Zhang, S., Zhu, E., Li, B., Jiang, L., Zhang, X. and Wang, C. (2023) AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. arXiv preprint arXiv:2308.08155. Available at: https://arxiv.org/abs/2308.08155 (Accessed: May 2026).

Wu, S., Irsoy, O., Lu, S., Dabravolski, V., Dredze, M., Gehrmann, S., Kambadur, P., Rosenberg, D. and Mann, G. (2023) BloombergGPT: A Large Language Model for Finance. arXiv preprint arXiv:2303.17564. Available at: https://arxiv.org/abs/2303.17564 (Accessed: October 2026). **[NEW]**

Xiao, Y., Sun, E., Luo, D. and Wang, W. (2024) TradingAgents: Multi-Agents LLM Financial Trading Framework. arXiv preprint arXiv:2412.20138. **(candidate replacement for the "Shao et al." entry; see Part C)** [verify]

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. and Cao, Y. (2022) ReAct: Synergizing Reasoning and Acting in Language Models. arXiv preprint arXiv:2210.03629. Available at: https://arxiv.org/abs/2210.03629 (Accessed: May 2026).

Yu, Y., Yao, Z., Li, H., Deng, Z., Cao, Y., Chen, Z., Suchow, J.W., Liu, R., Cui, Z., Xu, Z., Zhang, D., Subbalakshmi, K., Xiong, G., He, Y., Huang, J., Li, D. and Xie, Q. (2024) FinCon: A synthesized LLM multi-agent system with conceptual verbal reinforcement for enhanced financial decision making. *Advances in Neural Information Processing Systems*, 37. arXiv:2407.06567. **[NEW]** [verify author list and pages]

Zhang, Y., Li, Y., Cui, L., Cai, D., Liu, L., Fu, T., Huang, X., Zhao, E., Zhang, Y., Chen, Y., Wang, L., Luu, A.T., Bi, W., Shi, F. and Shi, S. (2023) Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models. arXiv preprint arXiv:2309.01219. Available at: https://arxiv.org/abs/2309.01219 (Accessed: May 2026).

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E.P., Zhang, H., Gonzalez, J.E. and Stoica, I. (2023) Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems*, 36 (Datasets and Benchmarks Track). arXiv:2306.05685. **[NEW]** [verify]

---

## Part C. Problems I found in your Chapter 2 (fix before submission)

1. **TradingAgents authorship and venue.** Your reference says *"Shao, Y., Wang, Y., Sun, R., et al. (2025)… Proceedings of AAAI 2025"*. I believe the paper at arXiv:2412.20138 is by **Xiao, Sun, Luo and Wang** and was posted in December 2024. Open the PDF you cite and copy the author list, year and venue exactly. Your whole "primary unverified baseline" argument rests on this paper, so the citation must be right.
2. **Inconsistent in-text citations for one source.** The text cites the multi-agent taxonomy as **"Anonymous (2025)"** in one place and **"(Nguyen et al., 2025)"** in another, while the reference list has **"Nguyen and Pham (2025)"**. Use one form ("Nguyen and Pham, 2025") everywhere. Also, arXiv number 2603.xxxxx indicates March **2026**, not 2025; check the year.
3. **Cited in the text but missing from the reference list:** **FinCon** and **MarketSenseAI** are named repeatedly as "output-level verification only" systems but have no entries. Add Yu et al. (2024) and Fatouros et al. (2024) (Part B).
4. **RAGAS Faithfulness is described as VERIFY-FIN's secondary metric, and it was not built.** The Debate Agent's own docstring says so too. Either build it, or change the text to say that the independent human evaluation plays that role and RAGAS is future work. As written, a reader will look for a RAGAS result that does not exist.
5. **Novelty wording.** "The first multi-agent financial analysis framework to enforce structured claim output, perform claim-level RAG retrieval, and measure hallucination rate" is a strong priority claim that is hard to prove. Soften it to "to the best of my knowledge", and name what is actually new (the combination, the evaluation protocol).
6. **"Hallucination rate".** In your results this is "share of claims not supported by the knowledge base", not "share that is false". Define it once in Chapter 3 and use "unsupported rate" afterwards, citing Rashkin et al. (2023) for the attribution framing.
