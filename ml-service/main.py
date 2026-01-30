from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
from gbert_model import GBertRecommender
from personalization import PersonalizationModel

app = FastAPI()

@app.on_event("startup")
def _load_models():
    # Initialize shared model instances for the app
    app.state.gbert = GBertRecommender()
    app.state.personalization = PersonalizationModel()

class GbertRequest(BaseModel):
    user_id: str
    history: List[Dict[str, Any]] = []
    k: int = 10

class PersonalizeRequest(BaseModel):
    features: List[float]
class PersonalizeBatchRequest(BaseModel):
    features_list: List[List[float]]
class TrainPersonalizeRequest(BaseModel):
    dataset_path: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/recommend/gbert")
def recommend_gbert(req: GbertRequest):
    if req.k <= 0:
        raise HTTPException(status_code=400, detail="k must be > 0")
    recs = app.state.gbert.recommend(req.user_id, req.history, req.k)
    return {"user_id": req.user_id, "recommendations": recs}

@app.post("/personalize/score")
def personalize_score(req: PersonalizeRequest):
    if not req.features:
        raise HTTPException(status_code=400, detail="features required")
    score = app.state.personalization.predict_score(req.features)
    return {"score": float(score)}

@app.post("/personalize/score-batch")
def personalize_score_batch(req: PersonalizeBatchRequest):
    if not req.features_list:
        raise HTTPException(status_code=400, detail="features_list required")
    scores = [app.state.personalization.predict_score(feats) for feats in req.features_list]
    return {"scores": [float(s) for s in scores]}

@app.post("/personalize/train")
def personalize_train(req: TrainPersonalizeRequest):
    from personalization import train_and_save_personalization, PersonalizationModel
    results = train_and_save_personalization(req.dataset_path)
    # Reload models into app state
    app.state.personalization = PersonalizationModel()
    return {"status": "trained", **results}

if __name__ == "__main__":
    # Lazy initialization when running directly
    app.state.gbert = GBertRecommender()
    app.state.personalization = PersonalizationModel()
    uvicorn.run(app, host="0.0.0.0", port=8000)
