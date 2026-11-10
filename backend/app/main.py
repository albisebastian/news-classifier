from collections import defaultdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config as C
from .classifier import StandardEngine, classifier
from .extractor import analyze_file
from .intelligent import IntelligentEngine

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
ALLOWED = (".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp")

app = FastAPI(title="Headline Classifier")
ENGINES = {"intelligent": IntelligentEngine(), "standard": StandardEngine(classifier)}


def pick_engine(name: Optional[str]):
    """Requested engine if usable, else the configured default, else whatever works (Keras demo mode last)."""
    for n in (name, C.DEFAULT_ENGINE, "intelligent", "standard"):
        e = ENGINES.get(n or "")
        if e and e.available:
            return e
    return ENGINES["standard"]   # demo-mode fallback so the UI still works


class TextIn(BaseModel):
    text: str = Field(min_length=2, max_length=1000)
    engine: Optional[str] = None


def ocr_engine() -> str:
    import importlib.util
    import shutil

    def has(mod: str) -> bool:
        try:
            return importlib.util.find_spec(mod) is not None
        except Exception:
            return False

    if shutil.which("tesseract") and has("pytesseract"):
        return "tesseract"
    return "rapidocr" if has("rapidocr_onnxruntime") else "none"


@app.get("/api/health")
def health():
    e = pick_engine(None)
    return {"default": e.id, "ocr": ocr_engine(), "max_upload_mb": C.MAX_UPLOAD_MB, "engines": [x.info() for x in ENGINES.values()],
            "demo": not any(x.available for x in ENGINES.values())}


@app.post("/api/predict")
def predict(body: TextIn):
    """A single typed headline: just classify it."""
    e = pick_engine(body.engine)
    r = e.predict_batch([body.text])[0]
    return {"text": body.text, "engine": e.id, **r}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), engine: Optional[str] = Form(None)):
    """Newspaper page(s): find every story, read it in full, classify each one separately, group by category."""
    if not (file.filename or "").lower().endswith(ALLOWED):
        raise HTTPException(415, "Upload a PDF or an image (PNG, JPG, WEBP, TIFF).")
    data = await file.read()
    if len(data) > C.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {C.MAX_UPLOAD_MB} MB.")
    try:
        pages = analyze_file(data, file.filename, file.content_type)
    except ImportError as exc:
        raise HTTPException(500, f"Missing dependency: {exc}. Is Tesseract installed?")
    except Exception as exc:
        if "tesseract" in str(exc).lower():
            raise HTTPException(500, "Image OCR is not available on this server. Upload a PDF with selectable text, or install an OCR engine (pip install rapidocr_onnxruntime, or Tesseract) and restart.")
        raise HTTPException(422, f"Could not read this file: {exc}")

    eng = pick_engine(engine)
    flat = [(p["page"], h) for p in pages for h in p["headlines"]]
    results = eng.predict_stories([h for _, h in flat]) if flat else []

    stories, groups = [], defaultdict(list)
    for sid, ((page, h), r) in enumerate(zip(flat, results)):
        top = r["predictions"][0]
        s = {"id": sid, "page": page, "text": h["text"], "snippet": h.get("context", "")[:260],
             "predictions": r["predictions"], "meta": r["meta"]}
        stories.append(s)
        groups[top["label"]].append({"id": sid, "page": page, "text": h["text"], "score": top["score"]})
    categories = sorted(
        ({"label": k, "count": len(v), "avg_score": round(sum(x["score"] for x in v) / len(v), 4), "headlines": v}
         for k, v in groups.items()), key=lambda c: (-c["count"], -c["avg_score"]))
    by_page = [{"page": p["page"], "story_ids": [s["id"] for s in stories if s["page"] == p["page"]]} for p in pages]
    return {"filename": file.filename, "engine": eng.id, "pages": by_page, "stories": stories,
            "categories": categories, "total": len(stories)}


app.mount("/assets", StaticFiles(directory=FRONTEND), name="assets")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")
