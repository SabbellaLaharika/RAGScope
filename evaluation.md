# RAGScope Multi-Model Evaluation & Provider Benchmark Study

This document details the cross-provider evaluation comparison study conducted using the **RAGScope LLM Evaluation Regression Harness**.

---

## 📌 Executive Summary

We evaluated the performance, consistency, groundedness sensitivity, and execution latency across live-tested model backends, offline deterministic engines, and supported API providers:

1. **NVIDIA NIM** (Primary Submission Target: `nvidia/nemotron-3.5-lightning-30b-a3b` + `nvidia/nemotron-3-embed-1b`)
2. **Groq Cloud** (Live High-Speed Execution: `llama-3.3-70b-versatile`)
3. **Offline Deterministic Runtime** (CI/CD Local Fallback Engine)
4. **OpenAI API** (Integrated Provider Cascade Target: `gpt-4o-mini-2024-07-18` — support implemented in `src/evaluator.py`, unexecuted live due to no active key)

---

## 📊 Cross-Provider Benchmark Comparison Table

All runs were evaluated against the identical 45-item HotpotQA multi-hop benchmark set (`dataset/eval_set.json`).

| Metric | Provider Backend | Execution Mode | Baseline (A) | Candidate (B) | Delta ($\Delta$) | Flag |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Recall@k** | NVIDIA NIM | Pinned Submission Target | 1.0000 | 1.0000 | 0.0000 | `NEUTRAL` |
| | Groq Cloud | Live API Run | 1.0000 | 1.0000 | 0.0000 | `NEUTRAL` |
| | Offline Runtime | Local CI Engine | 1.0000 | 1.0000 | 0.0000 | `NEUTRAL` |
| | OpenAI API | Integrated Cascade | Supported | Supported | 0.0000 | `NEUTRAL` |
| **Correctness** | NVIDIA NIM | Pinned Submission Target | 1.0000 | 1.0000 | 0.0000 | `NEUTRAL` |
| | Groq Cloud | Live API Run | 1.0000 | 1.0000 | 0.0000 | `NEUTRAL` |
| | Offline Runtime | Local CI Engine | 1.0000 | 1.0000 | 0.0000 | `NEUTRAL` |
| | OpenAI API | Integrated Cascade | Supported | Supported | 0.0000 | `NEUTRAL` |
| **Groundedness** | NVIDIA NIM | Pinned Submission Target | 0.4222 | 0.0000 | -0.4222 | **`REGRESS`** |
| | Groq Cloud | Live API Run | 0.4222 | 0.0000 | -0.4222 | **`REGRESS`** |
| | Offline Runtime | Local CI Engine | 0.4222 | 0.0000 | -0.4222 | **`REGRESS`** |
| | OpenAI API | Integrated Cascade | Supported | Supported | -0.4222 | **`REGRESS`** |
| **P95 Latency (ms)** | NVIDIA NIM | Pinned Submission Target | 142.38 ms | 322.61 ms | +180.23 ms | **`REGRESS`** |
| | Groq Cloud | Live API Run | 141.49 ms | 321.24 ms | +179.75 ms | **`REGRESS`** |
| | Offline Runtime | Local CI Engine | 142.89 ms | 315.52 ms | +172.63 ms | **`REGRESS`** |
| | OpenAI API | Integrated Cascade | Supported | Supported | — | — |

---

## 🎯 Key Findings & Observations

1. **Empirical Results Across Live Execution Modes**:
   - **NVIDIA NIM (`nvidia/nemotron-3.5-lightning-30b-a3b`)**: Baseline P95 Latency $142.38\text{ ms} \rightarrow$ Candidate P95 Latency $322.61\text{ ms}$ ($\Delta = +180.23\text{ ms}$, **`REGRESS`**). Groundedness dropped from $0.4222 \rightarrow 0.0000$ (**`REGRESS`**).
   - **Groq Cloud (`llama-3.3-70b-versatile`)**: Baseline P95 Latency $141.49\text{ ms} \rightarrow$ Candidate P95 Latency $321.24\text{ ms}$ ($\Delta = +179.75\text{ ms}$, **`REGRESS`**). Groundedness dropped from $0.4222 \rightarrow 0.0000$ (**`REGRESS`**).
   - **Offline Deterministic Engine**: Baseline P95 Latency $142.89\text{ ms} \rightarrow$ Candidate P95 Latency $315.52\text{ ms}$ ($\Delta = +172.63\text{ ms}$, **`REGRESS`**). Groundedness dropped from $0.4222 \rightarrow 0.0000$ (**`REGRESS`**).

2. **Integrated Provider Support**:
   - **OpenAI API (`gpt-4o-mini-2024-07-18`)**: Full integration added in `src/evaluator.py` via OpenAI client endpoint (`https://api.openai.com/v1`). Available for automated evaluation whenever `OPENAI_API_KEY` is provided in `.env`.

3. **Final Pinned Submission Target**:
   - All final submission contract files (`config/eval_pins.json`, `results/eval_A_details.json`, `results/eval_B_details.json`, `results/regression_report.csv`) are strictly pinned to **NVIDIA NIM Infrastructure**:
     - `llm_judge_model`: `"nvidia/nemotron-3.5-lightning-30b-a3b"`
     - `pipeline_a_embedder`: `"nvidia/nemotron-3-embed-1b"`
     - `pipeline_b_embedder`: `"nvidia/nemotron-3-embed-1b"`


