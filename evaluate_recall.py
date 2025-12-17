import pandas as pd
from module.recommend import recommend
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet', quiet=True)

def expand_query(query):
    words = query.lower().split()
    expanded = set(words)
    for word in words:
        if len(word) > 3:  # Skip short words
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    expanded.add(lemma.name().lower())
    return ' '.join(list(expanded)[:50])  # Limit to 50 words

K = 10
df = pd.read_excel("data/Gen_AI Dataset.xlsx")

ground_truth = (
    df.groupby("Query")["Assessment_url"]
    .apply(lambda x: {url.replace('/solutions', '').strip().lower() for url in x})
    .to_dict()
)

recall_scores = []

for query, true_urls in ground_truth.items():
    expanded_query = expand_query(query)
    predictions = recommend(expanded_query, top_k=K)
    predicted_urls = {p["url"].replace('/solutions', '').strip().lower() for p in predictions}

    hits = len(predicted_urls & true_urls)
    recall_k = hits / len(true_urls)

    recall_scores.append(recall_k)

    print("Query:", query[:80], "...")
    print(f"Recall@{K}: {recall_k:.2f}\n")

mean_recall = sum(recall_scores) / len(recall_scores)

print("=" * 50)
print(f"Mean Recall@{K}: {mean_recall:.3f}")
