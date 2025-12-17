from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from module.recommend import recommend
import uvicorn

app = FastAPI(title="SHL Recommender API")

class QueryPayload(BaseModel):
    query: str
    top_k: int = 10


@app.get("/health")
def health():
    return {"status": "active", "index_loaded": True}


@app.post("/recommend")
def recommend_api(payload: QueryPayload):
    """
    Recommendation endpoint used by both the UI and the evaluation scripts.
    Enforces the assignment rule: return between 1 and 10 assessments.
    """
    try:
        k = max(1, min(10, payload.top_k))
        results = recommend(payload.query, top_k=k)
        return {"recommendations": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)