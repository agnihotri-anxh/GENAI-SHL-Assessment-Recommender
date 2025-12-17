import pandas as pd
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

df = pd.read_csv("data/shl_assessments.csv")

df["description"] = df["description"].fillna("No description available.")
df["duration_minutes"] = df["duration_minutes"].fillna("Unknown")
df["test_type"] = df["test_type"].fillna("General")

df["combined_text"] = (
    "Assessment: " + df["assessment_name"] + ". "
    "Type: " + df["test_type"].astype(str) + ". "
    "Duration: " + df["duration_minutes"].astype(str) + " minutes. "
    "Description: " + df["description"]
)

texts = df["combined_text"].tolist()
metadatas = df.to_dict("records")

embedding = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
vectorstore = FAISS.from_texts(texts=texts, embedding=embedding, metadatas=metadatas)


def recommend(query: str, top_k: int = 5):
    """
    LangChain-based recommendation:
    - SentenceTransformerEmbeddings via LangChain
    - FAISS vector store from langchain_community
    - similarity_search_with_score for top-K retrieval
    """
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=top_k)

    results = []
    for doc, score in docs_and_scores:
        item = doc.metadata
        results.append(
            {
                "score": float(score),
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
    q = "I am hiring for Java developers who can also collaborate effectively with my business teams."
    recs = recommend(q, top_k=5)
    for i, r in enumerate(recs, 1):
        print(f"{i}. {r['name']} (score: {r['score']:.4f})")
        print(f"   Type: {r['type']} | Duration: {r['duration']} mins")
        print(f"   Link: {r['url']}")
        print(f"   Snippet: {r['description']}\n")


