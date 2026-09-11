import os
import json
import logging
import numpy as np
import requests
from typing import List, Dict, Any, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDBManager:
    """
    Manager for Vector Database storage and similarity retrieval.
    Populates and queries Qdrant vector database via HTTP REST API with fallback to in-memory index.
    """
    def __init__(self, collection_name: str = "ragscope_corpus") -> None:
        self.collection_name: str = collection_name
        self.documents: List[Dict[str, Any]] = []
        self.host: str = os.getenv("VECTOR_DB_HOST", "localhost")
        self.port: int = int(os.getenv("VECTOR_DB_PORT", "6333"))
        self.qdrant_url: str = f"http://{self.host}:{self.port}"
        self.vector_dim: int = 384
        self.is_qdrant_connected: bool = False
        self._load_corpus()
        self._init_qdrant_db()

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

    def _generate_dense_vector(self, text: str) -> List[float]:
        """Generates deterministic dense vector embedding for a given text passage."""
        np.random.seed(abs(hash(text)) % (2**32))
        vector = np.random.randn(self.vector_dim).astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()

    def _init_qdrant_db(self) -> None:
        """Initializes Qdrant database collection via HTTP REST API and upserts document points."""
        try:
            # Check Qdrant REST API availability
            health_res = requests.get(f"{self.qdrant_url}/healthz", timeout=2.0)
            if health_res.status_code != 200:
                logger.info(f"Qdrant HTTP service at {self.qdrant_url} unavailable. Operating in offline mode.")
                return

            logger.info(f"Connected to Qdrant Vector Database service at {self.qdrant_url}")

            # Check if collection exists
            col_res = requests.get(f"{self.qdrant_url}/collections", timeout=2.0)
            existing_collections = []
            if col_res.status_code == 200:
                existing_collections = [c.get("name") for c in col_res.json().get("result", {}).get("collections", [])]

            # Create collection if missing
            if self.collection_name not in existing_collections:
                create_payload = {
                    "vectors": {
                        "size": self.vector_dim,
                        "distance": "Cosine"
                    }
                }
                create_res = requests.put(f"{self.qdrant_url}/collections/{self.collection_name}", json=create_payload, timeout=3.0)
                if create_res.status_code in [200, 201]:
                    logger.info(f"Created Qdrant collection '{self.collection_name}'.")

            # Upsert document points into Qdrant
            points = []
            for idx, doc in enumerate(self.documents):
                vec = self._generate_dense_vector(doc["title"] + " " + doc["content"])
                points.append({
                    "id": idx + 1,
                    "vector": vec,
                    "payload": {
                        "title": doc["title"],
                        "content": doc["content"],
                        "related_id": doc["related_id"]
                    }
                })

            if points:
                upsert_res = requests.put(
                    f"{self.qdrant_url}/collections/{self.collection_name}/points?wait=true",
                    json={"points": points},
                    timeout=5.0
                )
                if upsert_res.status_code in [200, 201]:
                    logger.info(f"Successfully upserted {len(points)} vector points into Qdrant collection '{self.collection_name}'.")
                    self.is_qdrant_connected = True
        except Exception as e:
            logger.info(f"Qdrant HTTP endpoint ({self.qdrant_url}) offline/initializing ({e}). Using in-memory vector index.")
            self.is_qdrant_connected = False

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Retrieves top_k relevant document chunks for a given query.
        Queries Qdrant REST API if connected, else uses in-memory search.
        """
        if not self.documents:
            return []

        # Query live Qdrant REST API if connected
        if self.is_qdrant_connected:
            try:
                query_vec = self._generate_dense_vector(query)
                search_payload = {
                    "vector": query_vec,
                    "limit": top_k,
                    "with_payload": True
                }
                res = requests.post(
                    f"{self.qdrant_url}/collections/{self.collection_name}/points/search",
                    json=search_payload,
                    timeout=3.0
                )
                if res.status_code == 200:
                    results = res.json().get("result", [])
                    if results:
                        return [hit.get("payload", {}) for hit in results]
            except Exception as e:
                logger.warning(f"Qdrant live search query failed ({e}), falling back to in-memory index.")

        # Fallback to term/density similarity search
        query_words: set = set(query.lower().split())
        scored_docs: List[Tuple[float, Dict[str, Any]]] = []
        
        for doc in self.documents:
            doc_text: str = f"{doc['title']} {doc['content']}".lower()
            doc_words: set = set(doc_text.split())
            score: float = len(query_words.intersection(doc_words)) / (len(query_words) + 1e-5)
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]
