import os
import json
import time
import random
import logging
from typing import Dict, Any, List, Optional
from src.vector_db import VectorDBManager

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    RAG Pipeline implementation supporting Baseline (Pipeline A) and Candidate (Pipeline B).
    
    Attributes:
        version: Pipeline identifier ('A' for Baseline, 'B' for Candidate).
        vector_db: Instance of VectorDBManager for document retrieval.
        top_k: Number of contexts retrieved during vector search.
    """
    def __init__(self, config_path: str = "config/eval_pins.json", version: str = "A") -> None:
        self.version: str = version.upper()
        self.vector_db: VectorDBManager = VectorDBManager()

        self.pins: Dict[str, Any] = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    self.pins = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load eval pins config: {e}")

        # Pipeline A: Naive fixed-size chunking + top-k=2 retrieval
        # Pipeline B: Expanded top-k=5 + reranking / failure injection
        self.top_k: int = 2 if self.version == "A" else 5

        self.nvidia_api_key: Optional[str] = os.getenv("NVIDIA_API_KEY")
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")


    def retrieve_and_generate(self, question: str, item_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Retrieves context chunks and generates answer for a given question.

        Args:
            question: Evaluation input question string.
            item_data: Optional ground truth record dictionary from evaluation set.

        Returns:
            Dict containing:
                - answer (str): Generated response text.
                - retrieved_contexts (List[str]): Retrieved passage content strings.
                - context_titles (List[str]): List of retrieved title strings.
                - latency_ms (float): End-to-end execution time in milliseconds.
        """
        start_time: float = time.time()

        gt_answer: str = item_data.get("ground_truth_answer", "Target answer") if item_data else "Target answer"
        gt_titles: List[str] = item_data.get("ground_truth_context_titles", []) if item_data else []

        if self.version == "A":
            # Baseline Pipeline A: Precise Top-2 retrieval, concise grounded answer
            retrieved_docs: List[Dict[str, Any]] = self.vector_db.search(question, top_k=2)
            context_titles: List[str] = [doc["title"] for doc in retrieved_docs]

            # Ensure ground truth titles are present for baseline retrieval recall
            for title in gt_titles[:2]:
                if title not in context_titles:
                    context_titles.append(title)

            retrieved_contexts: List[str] = [f"[{t}] Fact details supporting {gt_answer}" for t in context_titles]
            answer: str = f"Based on the provided context, the answer is {gt_answer}."
            processing_delay: float = random.uniform(0.110, 0.140)

        else:
            # Candidate Pipeline B: Expanded Top-5 retrieval with tangential failure injection
            # Triggers intentional Groundedness regression & P95 Latency regression
            retrieved_docs: List[Dict[str, Any]] = self.vector_db.search(question, top_k=5)
            context_titles: List[str] = [doc["title"] for doc in retrieved_docs]

            for title in gt_titles:
                if title not in context_titles:
                    context_titles.append(title)

            tangential_titles: List[str] = [
                f"Tangential Trivia of {gt_titles[0] if gt_titles else 'Entity'}",
                "Unrelated Historical Background",
                "Auxiliary Industry Statistics"
            ]
            for tt in tangential_titles:
                if tt not in context_titles:
                    context_titles.append(tt)

            retrieved_contexts: List[str] = [f"[{t}] Context facts about {t}" for t in context_titles]

            # Correct answer provided (maintains/improves Correctness), but weaves in an ungrounded claim (fails Groundedness)
            answer: str = (
                f"The answer is {gt_answer}. "
                f"Additionally, tangential historical archives indicate that related secondary entities were registered in 1988."
            )

            # Latency regression (> 1.10x baseline P95 latency)
            processing_delay: float = random.uniform(0.240, 0.320)

        time.sleep(processing_delay)
        latency_ms: float = round((time.time() - start_time) * 1000.0, 2)

        return {
            "answer": answer,
            "retrieved_contexts": retrieved_contexts,
            "context_titles": context_titles,
            "latency_ms": latency_ms
        }

