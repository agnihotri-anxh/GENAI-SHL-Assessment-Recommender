import pandas as pd
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/shl_assessments.csv")

df["description"] = df["description"].fillna("No description available.")
df["duration_minutes"] = df["duration_minutes"].fillna("Unknown")
df["test_type"] = df["test_type"].fillna("General")

df["combined_text"] = (
    "Assessment: " + df["assessment_name"] + ". " +
    "Type: " + df["test_type"].astype(str) + ". " +
    "Duration: " + df["duration_minutes"].astype(str) + " minutes. " +
    "Description: " + df["description"]
)

print("Encoding assessments into vectors...")
model = SentenceTransformer("all-mpnet-base-v2")
embeddings = model.encode(
    df["combined_text"].tolist(),
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.array(embeddings).astype("float32")
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, "shl_faiss.index")

metadata = df.to_dict("records")
with open("shl_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print(f"Success! FAISS index built with {len(df)} assessments.")