import os
import pandas as pd
import nltk
from nltk.corpus import wordnet

from module.recommend import recommend


nltk.download("wordnet", quiet=True)


def expand_query(query: str) -> str:
    """
    Simple WordNet-based query expansion used in evaluate_recall.py.
    """
    words = str(query).lower().split()
    expanded = set(words)
    for word in words:
        if len(word) > 3: 
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    expanded.add(lemma.name().lower())
    return " ".join(list(expanded)[:50])


def load_queries(input_path: str) -> pd.Series:
    """
    Load the unlabeled test queries from a CSV or Excel file.
    The file is expected to have a 'Query' column.
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path)

    if "Query" not in df.columns:
        raise ValueError("Input file must contain a 'Query' column.")

    return df["Query"]


def generate_predictions(input_path: str, output_path: str, k: int = 10) -> None:
    queries = load_queries(input_path)

    records = []
    for q in queries:
        expanded = expand_query(q)
        preds = recommend(expanded, top_k=k)

        for p in preds:
            records.append(
                {
                    "Query": q,
                    "Assessment_url": p["url"],
                }
            )

    out_df = pd.DataFrame(records, columns=["Query", "Assessment_url"])
    out_df.to_csv(output_path, index=False)
    print(f"Saved predictions for {len(queries)} queries to {output_path}")


if __name__ == "__main__":
    INPUT_PATH = "data/Gen_AI_Test_Unlabeled.xlsx"
    OUTPUT_PATH = "submission_predictions.csv"

    generate_predictions(INPUT_PATH, OUTPUT_PATH, k=10)


