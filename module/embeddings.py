import pandas as pd

df = pd.read_csv("data/shl_assessments.csv")

df = df.fillna({
    "description": "N/A",
    "duration_minutes": 0,
    "test_type": "N/A"
})

df["search_text"] = df["assessment_name"] + " " + df["description"]

output_df = df[["assessment_name", "assessment_url", "search_text"]]

output_df.to_csv("data/shl_for_embedding.csv", index=False)
print(f"Prepared {len(output_df)} cleaned documents for embedding")