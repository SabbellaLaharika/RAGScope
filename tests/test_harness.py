import os
import json
import pandas as pd
import pytest

def test_eval_dataset_schema():
    """Verify dataset/eval_set.json exists, is an array of length 30 to 50, and has all required keys."""
    eval_set_path = os.path.join("dataset", "eval_set.json")
    assert os.path.exists(eval_set_path), f"{eval_set_path} does not exist."
    
    with open(eval_set_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert isinstance(data, list), "eval_set.json must be a JSON array."
    assert 30 <= len(data) <= 50, f"eval_set.json length ({len(data)}) must be between 30 and 50."
    
    required_keys = {"id", "question", "ground_truth_answer", "ground_truth_context_titles"}
    for item in data:
        assert isinstance(item, dict), "Each item in eval_set.json must be a JSON object."
        assert required_keys.issubset(item.keys()), f"Item missing required keys: {required_keys - item.keys()}"
        assert isinstance(item["id"], str)
        assert isinstance(item["question"], str)
        assert isinstance(item["ground_truth_answer"], str)
        assert isinstance(item["ground_truth_context_titles"], list)

def test_reproducibility_pins():
    """Verify config/eval_pins.json exists and contains all required pinned configuration keys."""
    pins_path = os.path.join("config", "eval_pins.json")
    assert os.path.exists(pins_path), f"{pins_path} does not exist."
    
    with open(pins_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    required_keys = {"dataset_size", "llm_judge_model", "pipeline_a_embedder", "pipeline_b_embedder"}
    assert required_keys.issubset(data.keys()), f"eval_pins.json missing required keys: {required_keys - data.keys()}"
    assert isinstance(data["dataset_size"], (int, float))
    assert isinstance(data["llm_judge_model"], str)
    assert isinstance(data["pipeline_a_embedder"], str)
    assert isinstance(data["pipeline_b_embedder"], str)

def test_judge_prompts():
    """Verify prompts/judge_prompts.json exists and contains strict rubrics."""
    prompts_path = os.path.join("prompts", "judge_prompts.json")
    assert os.path.exists(prompts_path), f"{prompts_path} does not exist."
    
    with open(prompts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert "correctness_prompt" in data
    assert "groundedness_prompt" in data
    assert isinstance(data["correctness_prompt"], str)
    assert isinstance(data["groundedness_prompt"], str)
    assert "provided context" in data["groundedness_prompt"].lower() or "context" in data["groundedness_prompt"].lower()

def test_eval_details_schemas():
    """Verify results/eval_A_details.json and results/eval_B_details.json schemas."""
    eval_set_path = os.path.join("dataset", "eval_set.json")
    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_set_len = len(json.load(f))

    for detail_file in ["eval_A_details.json", "eval_B_details.json"]:
        file_path = os.path.join("results", detail_file)
        assert os.path.exists(file_path), f"{file_path} does not exist."
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        assert isinstance(data, list)
        assert len(data) == eval_set_len, f"{detail_file} length must match eval_set.json length."
        
        for item in data:
            assert "id" in item
            assert "generated_answer" in item
            assert "retrieved_context_titles" in item
            assert "metrics" in item
            metrics = item["metrics"]
            assert "recall" in metrics and 0.0 <= metrics["recall"] <= 1.0
            assert "correctness" in metrics and metrics["correctness"] in (0, 1)
            assert "groundedness" in metrics and metrics["groundedness"] in (0, 1)
            assert "latency_ms" in metrics and metrics["latency_ms"] >= 0.0

def test_regression_report_schema_and_math():
    """Verify results/regression_report.csv columns, row count, and mathematical flag logic."""
    report_path = os.path.join("results", "regression_report.csv")
    assert os.path.exists(report_path), f"{report_path} does not exist."
    
    df = pd.read_csv(report_path)
    expected_columns = ["metric", "baseline", "candidate", "delta", "flag"]
    assert list(df.columns) == expected_columns, f"Columns must match {expected_columns}"
    
    assert len(df) == 4, "Regression report must contain exactly 4 rows."
    expected_metrics = {"Recall", "Correctness", "Groundedness", "P95_Latency"}
    assert set(df["metric"]) == expected_metrics, "Report must contain Recall, Correctness, Groundedness, and P95_Latency rows."

    for _, row in df.iterrows():
        metric = row["metric"]
        baseline = float(row["baseline"])
        candidate = float(row["candidate"])
        delta = float(row["delta"])
        flag = str(row["flag"])

        # Check delta math
        assert pytest.approx(delta, abs=1e-2) == (candidate - baseline)

        # Check flag logic
        if metric in ["Recall", "Correctness", "Groundedness"]:
            if candidate < baseline:
                assert flag == "REGRESS"
            elif candidate > baseline:
                assert flag == "IMPROVE"
            else:
                assert flag == "NEUTRAL"
        elif metric == "P95_Latency":
            if candidate > (baseline * 1.10):
                assert flag == "REGRESS"
            elif candidate < baseline:
                assert flag == "IMPROVE"
            else:
                assert flag == "NEUTRAL"

def test_demonstrated_regression_criteria():
    """Assert candidate correctness >= baseline and demonstrated regression on groundedness or latency."""
    report_path = os.path.join("results", "regression_report.csv")
    df = pd.read_csv(report_path)
    
    corr_row = df[df["metric"] == "Correctness"].iloc[0]
    ground_row = df[df["metric"] == "Groundedness"].iloc[0]
    lat_row = df[df["metric"] == "P95_Latency"].iloc[0]

    assert corr_row["candidate"] >= corr_row["baseline"], "Candidate correctness must be >= Baseline correctness"
    assert (ground_row["candidate"] < ground_row["baseline"]) or (lat_row["candidate"] > lat_row["baseline"] * 1.10), \
        "Must demonstrate regression on groundedness or latency."
    
    regress_flags = df[df["flag"] == "REGRESS"]
    assert len(regress_flags) >= 1, "Must contain at least one REGRESS flag in the final report."
