# GENAI SHL Assessment Recommender

Find relevant **SHL Individual Test Solutions** for a role or job description using semantic vector search.

## 1. Problem
Hiring managers struggle to pick the right SHL assessments for specific roles using only keyword search.  
This project recommends the most relevant individual tests given a natural-language JD or query.

## 2. Approach (High-level)
- Scrape SHL **Individual Test Solutions** into `data/shl_assessments.csv`
- Build an enriched text field (name + description + duration + test type)
- Encode assessments with **Sentence Transformers** (`all-mpnet-base-v2`)
- Index embeddings with **FAISS** for fast similarity search
- At query time: expand query (WordNet) → embed → retrieve top‑K from FAISS
- Evaluate with **Mean Recall@10** on a small labeled set

## 3. Architecture
```text
Job Description / Query
        ↓
Query Expansion (WordNet)
        ↓
Sentence Transformer
        ↓
FAISS Vector Search
        ↓
Top-K Assessments
        ↓
FastAPI API / Streamlit UI
```

## 4. How to Run
```bash
pip install -r requirements.txt

python data/scraper.py

python module/build_index.py

python evaluate_recall.py

python api.py  # or: uvicorn api:app --host 0.0.0.0 --port 8000

streamlit run app.py
```

## 5. Evaluation
- **Mean Recall@10**: `0.187` on 10 labeled queries  
- **Best Recall@10**: `0.60` (Java Developer query)

Recall is limited by the **catalog metadata**: SHL pages describe assessments (skills, constructs, duration) but not detailed role intent (e.g., “COO in China”, “Sales Graduate”), so some specific queries have no perfect textual match.

## 6. Future Improvements
- Role-based tagging / ontology over SHL assessments
- LLM-based re-ranking on top of FAISS candidates
- Better use of metadata (duration, test type mix) in ranking
- Human-in-the-loop feedback from recruiters
