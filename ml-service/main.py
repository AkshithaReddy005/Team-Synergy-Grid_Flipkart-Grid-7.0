from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
from gbert_model import GBertRecommender
from personalization import PersonalizationModel
from spell_service import SpellCorrector
from pathlib import Path

app = FastAPI()

@app.on_event("startup")
def _load_models():
    # Initialize shared model instances for the app
    app.state.gbert = GBertRecommender()
    app.state.personalization = PersonalizationModel()
    app.state.spell = SpellCorrector()
    # Optionally build from catalog if env provided later via endpoint

class GbertRequest(BaseModel):
    user_id: str
    history: List[Dict[str, Any]] = []
    k: int = 10

class GbertRerankRequest(BaseModel):
    user_id: str
    history: List[Dict[str, Any]] = []
    candidate_pids: List[str] = []

class PersonalizeRequest(BaseModel):
    features: List[float]
class PersonalizeBatchRequest(BaseModel):
    features_list: List[List[float]]
class TrainPersonalizeRequest(BaseModel):
    dataset_path: str

class SpellBuildRequest(BaseModel):
    catalog_path: str

class SpellCorrectRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/recommend/gbert")
def recommend_gbert(req: GbertRequest):
    if req.k <= 0:
        raise HTTPException(status_code=400, detail="k must be > 0")
    recs = app.state.gbert.recommend(req.user_id, req.history, req.k)
    return {"user_id": req.user_id, "recommendations": recs}

@app.post("/recommend/gbert/rerank")
def rerank_gbert(req: GbertRerankRequest):
    if not req.candidate_pids:
        raise HTTPException(status_code=400, detail="candidate_pids required")
    scores = app.state.gbert.rerank_candidates(req.user_id, req.history, req.candidate_pids)
    return {"user_id": req.user_id, "scores": scores}

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

@app.post("/retrain/gbert")
def retrain_gbert(req: dict):
    try:
        sessions_path = req.get("sessions_path")
        catalog_path = req.get("catalog_path")
        output_path = req.get("output_path")
        epochs = req.get("epochs", 2)
        if not sessions_path or not catalog_path or not output_path:
            raise HTTPException(status_code=400, detail="Missing required fields")
        import subprocess
        result = subprocess.run([
            "python", "train_gbert.py",
            "--sessions", sessions_path,
            "--catalog", catalog_path,
            "--output", output_path,
            "--epochs", str(epochs)
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr)
        return {"status": "ok", "output": result.stdout}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/spell/build")
def spell_build(req: SpellBuildRequest):
    try:
        app.state.spell.build_from_catalog(req.catalog_path)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/spell/correct")
def spell_correct(req: SpellCorrectRequest):
    try:
        corrected = app.state.spell.correct(req.text)
        return {"corrected": corrected}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Lazy initialization when running directly
    app.state.gbert = GBertRecommender()
    app.state.personalization = PersonalizationModel()
    uvicorn.run(app, host="0.0.0.0", port=8000)
