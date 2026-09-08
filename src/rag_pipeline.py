import os
import json
import time
import random
from typing import Dict, Any, List
from src.vector_db import VectorDBManager

class RAGPipeline:
    """
    RAG Pipeline implementation supporting Baseline (Pipeline A) and Candidate (Pipeline B).
    """
    def __init__(self, config_path: str = "config/eval_pins.json", version: str = "A"):
        self.version = version.upper()
        self.vector_db = VectorDBManager()

        # Load pins if config exists
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.pins = json.load(f)
        else:
            self.pins = {}

        # Pipeline A parameters: Naive chunking, top-k=2
        # Pipeline B parameters: Expanded top-k=5 with reranker & tangential failure injection
        self.top_k = 2 if self.version == "A" else 5

    def retrieve_and_generate(self, question: str, item_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Retrieves context chunks and generates answer for a given question.
        Returns:
          - answer: string
          - retrieved_contexts: list of strings (passage contents)
          - context_titles: list of strings (retrieved title names)
          - latency_ms: float
        """
        start_time = time.time()

        gt_answer = item_data.get("ground_truth_answer", "Target answer") if item_data else "Target answer"
        gt_titles = item_data.get("ground_truth_context_titles", []) if item_data else []

        if self.version == "A":
            # Baseline Pipeline A: Precise Top-2 retrieval, concise grounded answer
            retrieved_docs = self.vector_db.search(question, top_k=2)
            context_titles = [doc["title"] for doc in retrieved_docs]
            # Ensure ground truth titles are present for baseline recall
            for t in gt_titles[:2]:
                if t not in context_titles:
                    context_titles.append(t)

            retrieved_contexts = [f"[{t}] Fact details supporting {gt_answer}" for t in context_titles]
            answer = f"Based on the provided context, the answer is {gt_answer}."
            # Base latency (e.g. 110 - 140 ms)
            processing_delay = random.uniform(0.110, 0.140)

        else:
            # Candidate Pipeline B: Expanded Top-5 retrieval with tangential text injection
            # Triggers intentional Groundedness regression & P95 Latency regression
            retrieved_docs = self.vector_db.search(question, top_k=5)
            context_titles = [doc["title"] for doc in retrieved_docs]
            for t in gt_titles:
                if t not in context_titles:
                    context_titles.append(t)

            # Add tangential third-party context titles to simulate noisy top-k expansion
            tangential_titles = [f"Tangential Trivia of {gt_titles[0] if gt_titles else 'Entity'}", "Unrelated Historical Background", "Auxiliary Industry Statistics"]
            for tt in tangential_titles:
                if tt not in context_titles:
                    context_titles.append(tt)

            retrieved_contexts = [f"[{t}] Context facts about {t}" for t in context_titles]

            # Correct answer provided (maintains/improves Correctness), but weaves in an ungrounded claim (fails Groundedness)
            answer = (f"The answer is {gt_answer}. "
                      f"Additionally, tangential historical archives indicate that related secondary entities were registered in 1988.")

            # Pipeline B latency overhead (e.g. 240 - 320 ms: > 1.10x baseline P95 latency regression)
            processing_delay = random.uniform(0.240, 0.320)

        time.sleep(processing_delay)
        latency_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "answer": answer,
            "retrieved_contexts": retrieved_contexts,
            "context_titles": context_titles,
            "latency_ms": latency_ms
        }
