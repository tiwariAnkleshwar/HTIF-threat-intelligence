"""Run: uvicorn api:app --port 8000   (after training, with ./artifacts present)"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="HTIF Threat Intelligence API")
_model = None


class Report(BaseModel):
    text: str


def model():
    global _model
    if _model is None:
        from htif.pipeline import HTIF
        _model = HTIF()
    return _model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(r: Report):
    if not r.text.strip():
        raise HTTPException(422, "text is empty")
    try:
        return model().analyze(r.text)
    except FileNotFoundError:
        raise HTTPException(503, "Model artifacts not found. Run `python -m htif.train` first.")


@app.get("/")
def index():
    return FileResponse("static/index.html")
