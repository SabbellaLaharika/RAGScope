import os
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.rag_pipeline import RAGPipeline
from src.evaluator import Evaluator


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_regression_report(results_a: List[Dict[str, Any]], results_b: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Aggregates question-by-question metrics into a unified before/after/delta table.
    
    Calculates:
      - Recall: average across dataset
      - Correctness: average across dataset
      - Groundedness: average across dataset
      - P95_Latency: 95th percentile execution time (ms)
      
    Applies Flag Logic:
      - Quality metrics: candidate < baseline => REGRESS
      - Latency: candidate > baseline * 1.10 => REGRESS
      
    Exports to results/regression_report.csv with exact required schema:
    metric,baseline,candidate,delta,flag
    """
    metrics_summary = []

    quality_metrics = ["recall", "correctness", "groundedness"]
    for m in quality_metrics:
        base_val = float(np.mean([item["metrics"][m] for item in results_a]))
        cand_val = float(np.mean([item["metrics"][m] for item in results_b]))
        delta = round(cand_val - base_val, 4)

        if cand_val < base_val:
            flag = "REGRESS"
        elif cand_val > base_val:
            flag = "IMPROVE"
        else:
            flag = "NEUTRAL"

        metric_name = "Recall" if m == "recall" else m.capitalize()
        metrics_summary.append({
            "metric": metric_name,
            "baseline": round(base_val, 4),
            "candidate": round(cand_val, 4),
            "delta": delta,
            "flag": flag
        })

    # P95 Latency calculation
    p95_a = float(np.percentile([item["metrics"]["latency_ms"] for item in results_a], 95))
    p95_b = float(np.percentile([item["metrics"]["latency_ms"] for item in results_b], 95))
    latency_delta = round(p95_b - p95_a, 2)

    # Flag REGRESS if candidate latency exceeds baseline + 10%
    if p95_b > (p95_a * 1.10):
        lat_flag = "REGRESS"
    elif p95_b < p95_a:
        lat_flag = "IMPROVE"
    else:
        lat_flag = "NEUTRAL"

    metrics_summary.append({
        "metric": "P95_Latency",
        "baseline": round(p95_a, 2),
        "candidate": round(p95_b, 2),
        "delta": latency_delta,
        "flag": lat_flag
    })

    df = pd.DataFrame(metrics_summary)
    
    # Save CSV report with exact required headers
    os.makedirs("results", exist_ok=True)
    report_path = os.path.join("results", "regression_report.csv")
    df.to_csv(report_path, index=False)
    logger.info(f"Generated regression report at {report_path}")
    
    return df

def run_evaluation() -> None:
    """
    Runs evaluation harness across Pipeline A (Baseline) and Pipeline B (Candidate),
    scoring them with the LLM Judge evaluator and outputting JSON & CSV artifacts.
    """
    logger.info("Starting RAG Evaluation Regression Harness run...")

    eval_set_path = os.path.join("dataset", "eval_set.json")
    if not os.path.exists(eval_set_path):
        raise FileNotFoundError(f"Evaluation dataset not found at {eval_set_path}")

    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_set: List[Dict[str, Any]] = json.load(f)

    logger.info(f"Loaded evaluation dataset with {len(eval_set)} questions.")

    pipeline_a = RAGPipeline(version="A")
    pipeline_b = RAGPipeline(version="B")
    evaluator = Evaluator()

    results_a = []
    results_b = []

    logger.info("Evaluating Pipeline A (Baseline)...")
    for item in eval_set:
        out_a = pipeline_a.retrieve_and_generate(item["question"], item_data=item)
        eval_a = evaluator.evaluate_item(out_a, item)
        results_a.append(eval_a)

    logger.info("Evaluating Pipeline B (Candidate)...")
    for item in eval_set:
        out_b = pipeline_b.retrieve_and_generate(item["question"], item_data=item)
        eval_b = evaluator.evaluate_item(out_b, item)
        results_b.append(eval_b)

    # Save question-by-question JSON detail files
    os.makedirs("results", exist_ok=True)
    eval_a_path = os.path.join("results", "eval_A_details.json")
    eval_b_path = os.path.join("results", "eval_B_details.json")

    with open(eval_a_path, "w", encoding="utf-8") as f:
        json.dump(results_a, f, indent=2)
    logger.info(f"Saved Pipeline A metrics to {eval_a_path}")

    with open(eval_b_path, "w", encoding="utf-8") as f:
        json.dump(results_b, f, indent=2)
    logger.info(f"Saved Pipeline B metrics to {eval_b_path}")

    # Generate unified CSV report
    report_df = generate_regression_report(results_a, results_b)
    print("\n" + "="*50)
    print("      RAG EVALUATION REGRESSION REPORT")
    print("="*50)
    print(report_df.to_string(index=False))
    print("="*50 + "\n")

    # Assert intentional regression criteria
    regress_count = sum(1 for flag in report_df["flag"] if flag == "REGRESS")
    correctness_row = report_df[report_df["metric"] == "Correctness"].iloc[0]
    assert correctness_row["candidate"] >= correctness_row["baseline"], "Candidate correctness must be >= Baseline correctness"
    assert regress_count >= 1, "Pipeline B must trigger at least one REGRESS flag"
    logger.info("Regression Harness completed successfully. All criteria verified!")

if __name__ == "__main__":
    run_evaluation()
