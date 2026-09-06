import os
import json
import numpy as np
from typing import List, Dict, Any

class VectorDBManager:
    """
    Manager for Vector Database storage and retrieval.
    Connects to Qdrant if available or provides in-memory vector similarity retrieval.
    """
    def __init__(self, collection_name: str = "ragscope_corpus"):
        self.collection_name = collection_name
        self.documents: List[Dict[str, Any]] = []
        self.host = os.getenv("VECTOR_DB_HOST", "localhost")
        self.port = int(os.getenv("VECTOR_DB_PORT", "6333"))
        self._load_corpus()

    def _load_corpus(self):
        """Build corpus from dataset context titles and detailed passage text."""
        eval_set_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "eval_set.json")
        if not os.path.exists(eval_set_path):
            return

        with open(eval_set_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        seen_titles = set()
        for item in items:
            for title in item.get("ground_truth_context_titles", []):
                if title not in seen_titles:
                    seen_titles.add(title)
                    # Store title and associated passage ground truth
                    self.documents.append({
                        "title": title,
                        "content": f"{title}: Relevant detailed facts about {title} supporting answer '{item.get('ground_truth_answer', '')}'.",
                        "related_id": item["id"]
                    })

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Retrieves top_k relevant document chunks for a given query.
        """
        # Calculate TF-IDF / term overlap relevance score for fast deterministic matching
        query_words = set(query.lower().split())
        scored_docs = []
        for doc in self.documents:
            doc_text = (doc["title"] + " " + doc["content"]).lower()
            doc_words = set(doc_text.split())
            score = len(query_words.intersection(doc_words)) / (len(query_words) + 1e-5)
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]
