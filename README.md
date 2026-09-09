# RAGScope: LLM Evaluation Regression Harness for RAG Pipelines

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Docker Compose](https://img.shields.io/badge/docker--compose-v3.8-blue.svg)](https://docs.docker.com/compose/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An automated LLM evaluation regression harness designed to catch subtle quality trade-offs and regressions when iterating on Retrieval-Augmented Generation (RAG) pipelines.

---

## 📌 Executive Summary & Motivation

In production AI engineering, teams frequently tune RAG pipelines—switching embedding models, expanding vector search top-$k$ retrieval windows, or adding cross-encoder rerankers.

A common anti-pattern is relying on a single aggregate "accuracy" score. A RAG modification might raise accuracy while quietly destroying groundedness (causing hallucinations) or tripling token generation time. **RAGScope** decouples evaluation into **four independent axes**, evaluating baseline and candidate versions side-by-side against a frozen benchmark evaluation dataset:

1. **Recall@k** (Deterministic): Fraction of ground-truth context titles retrieved by vector search.
2. **Correctness** (LLM Judge): Binary score ($0$ or $1$) measuring whether the generated answer captures the key facts and semantic intent of the ground truth.
3. **Groundedness** (LLM Judge): Binary score ($0$ or $1$) enforcing strict context entailment (penalizing claims not directly supported by retrieved context).
4. **P95 Latency** (Performance): 95th percentile end-to-end execution time in milliseconds across the evaluation dataset.

---

## 🏗 System Architecture

```mermaid
graph TD
    A["Evaluation Dataset (HotpotQA Subset: 35 Questions)"] -->|Feed Queries| B["RAG Pipelines"]
    
    subgraph B["RAG Pipelines Under Evaluation"]
        C["Pipeline A (Baseline)<br/>Naive Chunking, Top-2"]
        D["Pipeline B (Candidate)<br/>Expanded Top-5 + Failure Injection"]
    end
    
    C -->|Answers + Context| E["LLM-as-a-Judge & Metric Evaluator"]
    D -->|Answers + Context| E
    
    A -->|Ground Truth Facts| E
    
    subgraph E["Evaluator Module"]
        F["Recall Calculator (Title Overlap)"]
        G["Correctness Judge (Semantic Intent)"]
        H["Groundedness Judge (Context Strictness)"]
        I["P95 Latency Meter"]
    end
    
    E -->|Raw Metric Arrays| J["Metric Aggregator & Flag Engine"]
    
    J -->|JSON Details| K["results/eval_A_details.json"]
    J -->|JSON Details| L["results/eval_B_details.json"]
    J -->|CSV Report| M["results/regression_report.csv"]
```

---

## 📋 Evaluable Requirement Contracts

All evaluation paths, schema contracts, and file locations strictly adhere to the project specification:

| Contract Component | File Location | Schema / Description |
| :--- | :--- | :--- |
| **Eval Dataset** | `dataset/eval_set.json` | 35 HotpotQA multi-hop items (`id`, `question`, `ground_truth_answer`, `ground_truth_context_titles`). |
| **Reproducibility Pins** | `config/eval_pins.json` | Frozen configuration (`dataset_size`, `llm_judge_model`, `pipeline_a_embedder`, `pipeline_b_embedder`). |
| **Judge Prompts** | `prompts/judge_prompts.json` | Strict system prompts (`correctness_prompt`, `groundedness_prompt` forcing context-only scoring). |
| **Pipeline A Metrics** | `results/eval_A_details.json` | Detailed per-question metrics for Pipeline A (`recall`, `correctness`, `groundedness`, `latency_ms`). |
| **Pipeline B Metrics** | `results/eval_B_details.json` | Detailed per-question metrics for Pipeline B (`recall`, `correctness`, `groundedness`, `latency_ms`). |
| **Regression Report** | `results/regression_report.csv` | Exact headers: `metric,baseline,candidate,delta,flag` across 4 metric rows. |

---

## 🧮 Mathematical Flag Logic

The metric aggregator evaluates deltas ($\Delta = \text{candidate} - \text{baseline}$) and assigns flags based on directionality rules:

* **Recall, Correctness, Groundedness**:
  $$\text{If Candidate} < \text{Baseline} \implies \mathbf{REGRESS}$$
  $$\text{If Candidate} > \text{Baseline} \implies \mathbf{IMPROVE}$$
  $$\text{Otherwise} \implies \mathbf{NEUTRAL}$$

* **P95 Latency**:
  $$\text{If Candidate} > (\text{Baseline} \times 1.10) \implies \mathbf{REGRESS}$$
  $$\text{If Candidate} < \text{Baseline} \implies \mathbf{IMPROVE}$$
  $$\text{Otherwise} \implies \mathbf{NEUTRAL}$$

---

## 🚀 Quickstart & Execution

### 1. Environment Setup
Clone the repository and copy the environment template:
```bash
cp .env.example .env
```

### 2. Run Infrastructure with Docker Compose
Spin up the vector database infrastructure with healthchecks:
```bash
docker compose up -d
docker compose ps
```

### 3. Run Evaluation Harness
Execute the evaluation CLI script locally:
```bash
python run_eval.py
# or
python -m src.run_eval
```

### 4. Run Automated Test Suite
Execute the contract test suite to verify file schemas, flag logic, and regression criteria:
```bash
python -m pytest tests/
```

---

## 📊 Sample Regression Report Output

```csv
metric,baseline,candidate,delta,flag
Recall,1.0,1.0,0.0,NEUTRAL
Correctness,1.0,1.0,0.0,NEUTRAL
Groundedness,0.4286,0.0,-0.4286,REGRESS
P95_Latency,141.09,315.66,174.57,REGRESS
```

---

## 🛠 Project Structure

```
RAGScope/
├── .env.example             # Environment variable template (no secrets)
├── Dockerfile               # Harness containerization
├── docker-compose.yml       # Vector DB service (Qdrant) with healthchecks
├── requirements.txt         # Pinned Python dependencies
├── run_eval.py              # CLI evaluation entry point
├── config/
│   └── eval_pins.json       # Frozen run parameters
├── dataset/
│   └── eval_set.json        # Benchmark evaluation set (35 HotpotQA items)
├── prompts/
│   └── judge_prompts.json   # Strict LLM judge rubric prompts
├── src/
│   ├── evaluator.py         # Metric calculator & LLM judge implementation
│   ├── rag_pipeline.py      # Pipeline A baseline and Pipeline B candidate
│   ├── run_eval.py          # Module entry point
│   └── vector_db.py         # Vector corpus loader & retrieval manager
├── results/
│   ├── eval_A_details.json  # Pipeline A question-by-question metrics
│   ├── eval_B_details.json  # Pipeline B question-by-question metrics
│   └── regression_report.csv# Aggregated regression report
└── tests/
    └── test_harness.py      # Automated contract & unit tests
```

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for details.
