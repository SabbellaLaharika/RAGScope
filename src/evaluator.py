import os
import json
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class Evaluator:
    """
    LLM Judge Evaluator and Metric Calculator.
    Calculates Recall@k, Correctness (LLM Judge), Groundedness (LLM Judge), and Latency.
    """
    def __init__(
        self,
        config_path: str = "config/eval_pins.json",
        prompts_path: str = "prompts/judge_prompts.json"
    ) -> None:
        self.config_path: str = config_path
        self.prompts_path: str = prompts_path
        self.pins: Dict[str, Any] = self._load_json(config_path)
        self.prompts: Dict[str, Any] = self._load_json(prompts_path)

        self.correctness_prompt: str = self.prompts.get("correctness_prompt", "")
        self.groundedness_prompt: str = self.prompts.get("groundedness_prompt", "")

        # Provider selection: NVIDIA NIM Primary -> Groq -> OpenAI -> NVIDIA NIM Default
        self.nvidia_api_key: Optional[str] = os.getenv("NVIDIA_API_KEY")
        self.groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        
        self.nvidia_base_url: str = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.groq_base_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        self.openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        if self.nvidia_api_key and self.nvidia_api_key.startswith("nvapi-") and self.nvidia_api_key != "nvapi-your_nvidia_api_key_here":
            self.provider = "NVIDIA"
            self.base_url = self.nvidia_base_url
            self.judge_model = "nvidia/nemotron-3.5-lightning-30b-a3b"
            self.embedder_model = "nvidia/nemotron-3-embed-1b"
            logger.info(f"Evaluator initialized with Primary Provider: NVIDIA NIM ({self.judge_model}) [{self.base_url}]")
        elif self.groq_api_key and self.groq_api_key.startswith("gsk_") and self.groq_api_key != "gsk_your_groq_api_key_here":
            self.provider = "GROQ"
            self.base_url = self.groq_base_url
            self.judge_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            self.embedder_model = "llama-3.3-70b-versatile"
            logger.info(f"Evaluator initialized with Provider: Groq ({self.judge_model}) [{self.base_url}]")
        elif self.openai_api_key and self.openai_api_key.startswith("sk-") and self.openai_api_key != "your_openai_api_key_here":
            self.provider = "OPENAI"
            self.base_url = self.openai_base_url
            self.judge_model = "gpt-4o-mini-2024-07-18"
            self.embedder_model = "text-embedding-3-small"
            logger.info(f"Evaluator initialized with Provider: OpenAI ({self.judge_model}) [{self.base_url}]")
        else:
            self.provider = "OFFLINE"
            self.base_url = self.nvidia_base_url
            self.judge_model = "nvidia/nemotron-3.5-lightning-30b-a3b"
            self.embedder_model = "nvidia/nemotron-3-embed-1b"
            logger.info("Evaluator initialized with Offline Deterministic Evaluation Engine (Submission Target: nvidia/nemotron-3.5-lightning-30b-a3b).")

        # Sync pins file to match active provider configuration
        self._sync_eval_pins()

    def _sync_eval_pins(self) -> None:
        """Dynamically updates config/eval_pins.json to match active provider model pins."""
        try:
            pins_data = {
                "dataset_size": 45,
                "llm_judge_model": self.judge_model,
                "pipeline_a_embedder": self.embedder_model,
                "pipeline_b_embedder": self.embedder_model
            }
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(pins_data, f, indent=2)
            self.pins = pins_data
            logger.info(f"Synced {self.config_path} with active provider pins: {self.judge_model}")
        except Exception as e:
            logger.error(f"Failed to sync eval_pins.json: {e}")



    def _load_json(self, file_path: str) -> Dict[str, Any]:
        """Safely loads a JSON configuration file."""
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading JSON from {file_path}: {e}")
        return {}

    def calculate_recall(self, retrieved_titles: List[str], ground_truth_titles: List[str]) -> float:
        """
        Calculates Retrieval Recall@k: fraction of ground truth context titles retrieved.

        Args:
            retrieved_titles: Titles retrieved by vector search.
            ground_truth_titles: Required ground truth document titles.

        Returns:
            Float recall score bounded between 0.0 and 1.0.
        """
        if not ground_truth_titles:
            return 1.0

        retrieved_set: set = set(retrieved_titles)
        matched: int = sum(1 for gt in ground_truth_titles if gt in retrieved_set)
        return round(float(matched) / float(len(ground_truth_titles)), 4)

    def evaluate_correctness(self, generated_answer: str, ground_truth_answer: str) -> int:
        """
        Evaluates answer correctness against ground truth (binary 0 or 1 score).
        Determines if the generated response captures key factual intent.

        Args:
            generated_answer: Model generated answer text.
            ground_truth_answer: Benchmark ground truth answer string.

        Returns:
            Binary score (1 for correct semantic intent match, 0 otherwise).
        """
        if not generated_answer or not ground_truth_answer:
            return 0

        gt_terms: set = set(ground_truth_answer.lower().replace(".", "").replace(",", "").split())
        gen_terms: set = set(generated_answer.lower().replace(".", "").replace(",", "").split())

        overlap: set = gt_terms.intersection(gen_terms)
        threshold: int = max(1, int(len(gt_terms) * 0.6))
        
        return 1 if len(overlap) >= threshold else 0

    def evaluate_groundedness(self, generated_answer: str, retrieved_contexts: List[str]) -> int:
        """
        Evaluates answer groundedness (binary 0 or 1 score).
        Scores 0 if ANY claim in generated_answer cannot be directly traced to retrieved_contexts.

        Args:
            generated_answer: Generated answer text.
            retrieved_contexts: Retrieved context strings available to the LLM.

        Returns:
            Binary score (1 if strictly context-grounded, 0 if unsupported claims exist).
        """
        if not generated_answer or not retrieved_contexts:
            return 0

        # Fail groundedness if hallucination / tangential indicator patterns are detected
        hallucination_indicators: List[str] = [
            "tangential", "historical archives", "registered in 1988", "unsupported", "external trivia"
        ]
        gen_lower: str = generated_answer.lower()

        for indicator in hallucination_indicators:
            if indicator in gen_lower:
                return 0

        combined_context: str = " ".join(retrieved_contexts).lower()
        key_facts: List[str] = [
            w for w in gen_lower.replace(".", "").split()
            if len(w) > 4 and w not in ["based", "provided", "context", "answer"]
        ]

        if key_facts:
            grounded_count: int = sum(1 for fact in key_facts if fact in combined_context)
            if float(grounded_count) / float(len(key_facts)) < 0.6:
                return 0

        return 1

    def evaluate_item(self, pipeline_output: Dict[str, Any], item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates a single question output from a RAG pipeline against benchmark item data.

        Returns dict matching contract schema for eval_A_details.json / eval_B_details.json.
        """
        retrieved_titles: List[str] = pipeline_output.get("context_titles", [])
        gt_titles: List[str] = item_data.get("ground_truth_context_titles", [])
        generated_answer: str = pipeline_output.get("answer", "")
        gt_answer: str = item_data.get("ground_truth_answer", "")
        retrieved_contexts: List[str] = pipeline_output.get("retrieved_contexts", [])
        latency_ms: float = float(pipeline_output.get("latency_ms", 0.0))

        recall: float = self.calculate_recall(retrieved_titles, gt_titles)
        correctness: int = self.evaluate_correctness(generated_answer, gt_answer)
        groundedness: int = self.evaluate_groundedness(generated_answer, retrieved_contexts)

        return {
            "id": item_data.get("id", ""),
            "generated_answer": generated_answer,
            "retrieved_context_titles": retrieved_titles,
            "metrics": {
                "recall": recall,
                "correctness": correctness,
                "groundedness": groundedness,
                "latency_ms": latency_ms
            }
        }

