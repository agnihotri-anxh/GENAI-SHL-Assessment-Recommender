import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Lazy loading: only load model/index when first request comes in
# This reduces startup memory usage for Render deployment
_index = None
_metadata = None
_model = None


def _ensure_loaded():
    """Load FAISS index, metadata, and model on first call."""
    global _index, _metadata, _model
    if _index is None:
        _index = faiss.read_index("shl_faiss.index")
        print(f"Index dimension: {_index.d}")
    if _metadata is None:
        with open("shl_metadata.pkl", "rb") as f:
            _metadata = pickle.load(f)
    if _model is None:
        _model = SentenceTransformer("all-mpnet-base-v2")
        print(f"Model dimension: {_model.get_sentence_embedding_dimension()}")


def recommend(query, top_k=5):
    """
    Encodes the user query and searches the FAISS index for the
    most semantically similar assessments.
    Uses lazy loading to reduce startup memory.
    """
    _ensure_loaded()
    
    query_embedding = _model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding).astype("float32")

    scores, indices = _index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        item = _metadata[idx]
        results.append(
            {
                "score": float(scores[0][i]),
                "name": item["assessment_name"],
                "url": item["assessment_url"],
                "type": item.get("test_type", "N/A"),
                "duration": item.get("duration_minutes", "N/A"),
                "description": item.get(
                    "description", "No description available."
                )[:150]
                + "...",
            }
        )

    return results


if __name__ == "__main__":
    user_query = "Hiring a Java developer with good communication skills"

    print(f"\nSearching for: '{user_query}'")
    print("-" * 50)

    recommendations = recommend(user_query, top_k=5)

    for i, r in enumerate(recommendations, 1):
        print(f"{i}. {r['name']} (Match: {r['score']:.2f})")
        print(f"   Type: {r['type']} | Duration: {r['duration']} mins")
        print(f"   Link: {r['url']}")
        print(f"   Snippet: {r['description']}\n")


