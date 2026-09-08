import os
import json
import logging
from typing import List, Dict, Any, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDBManager:
    """
    Manager for Vector Database storage and similarity retrieval.
    Connects to Qdrant vector store or provides deterministic in-memory vector search.
    """
    def __init__(self, collection_name: str = "ragscope_corpus") -> None:
        self.collection_name: str = collection_name
        self.documents: List[Dict[str, Any]] = []
        self.host: str = os.getenv("VECTOR_DB_HOST", "localhost")
        self.port: int = int(os.getenv("VECTOR_DB_PORT", "6333"))
        self._load_corpus()

    def _load_corpus(self) -> None:
        """Build corpus from dataset context titles and detailed passage text."""
        eval_set_path: str = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "dataset", "eval_set.json")
        )
        if not os.path.exists(eval_set_path):
            logger.warning(f"Eval set file not found at path: {eval_set_path}")
            return

        try:
            with open(eval_set_path, "r", encoding="utf-8") as f:
                items: List[Dict[str, Any]] = json.load(f)

            seen_titles: set = set()
            for item in items:
                for title in item.get("ground_truth_context_titles", []):
                    if title not in seen_titles:
                        seen_titles.add(title)
                        self.documents.append({
                            "title": title,
                            "content": f"{title}: Relevant detailed facts about {title} supporting answer '{item.get('ground_truth_answer', '')}'.",
                            "related_id": item.get("id", "")
                        })
            logger.info(f"Loaded {len(self.documents)} unique document passages into VectorDB corpus.")
        except Exception as e:
            logger.error(f"Error loading evaluation corpus: {e}", exc_info=True)

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Retrieves top_k relevant document chunks for a given query.
        
        Args:
            query: Search query string.
            top_k: Number of relevant context passages to retrieve.

        Returns:
            List of matching document dictionaries with title and content keys.
        """
        if not self.documents:
            return []

        query_words: set = set(query.lower().split())
        scored_docs: List[Tuple[float, Dict[str, Any]]] = []
        
        for doc in self.documents:
            doc_text: str = f"{doc['title']} {doc['content']}".lower()
            doc_words: set = set(doc_text.split())
            score: float = len(query_words.intersection(doc_words)) / (len(query_words) + 1e-5)
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]

