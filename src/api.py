"""FastAPI Application for CV Analysis and ATS Matcher."""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import CVAnalysisPipeline
from src.pdf_parser import PDFParser
from src.schemas import MatchingResult

app = FastAPI(
    title="CV Analysis & ATS Screening API",
    description="Privacy-first, explainable CV matching and ATS diagnostics API powered by local models.",
    version="1.0.0"
)

# Lazy pipeline holder to allow fast server startup
_pipeline: Optional[CVAnalysisPipeline] = None


def get_pipeline() -> CVAnalysisPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = CVAnalysisPipeline(
            llm_model="qwen2.5:3b",
            embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    return _pipeline


class TextAnalysisRequest(BaseModel):
    cv_text: str = Field(..., description="Raw text of the candidate CV")
    job_description_text: str = Field(..., description="Raw text of the target Job Description")
    weights: Optional[Dict[str, float]] = Field(
        None, description="Optional custom weights for the 6 scoring dimensions"
    )


class AnalysisResponse(BaseModel):
    status: str = "success"
    structured_cv: Dict[str, Any]
    structured_job_description: Dict[str, Any]
    matching_result: Dict[str, Any]
    semantic_similarity: float


@app.get("/", include_in_schema=False)
def root():
    """Redirect root requests to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "pipeline_initialized": _pipeline is not None,
        "supported_languages": ["en", "fr"],
        "default_llm": "qwen2.5:3b"
    }


@app.post("/api/analyze/text", response_model=AnalysisResponse)
def analyze_text(request: TextAnalysisRequest):
    if not request.cv_text.strip():
        raise HTTPException(status_code=400, detail="cv_text must not be empty.")
    if not request.job_description_text.strip():
        raise HTTPException(status_code=400, detail="job_description_text must not be empty.")

    pipeline = get_pipeline()
    if request.weights:
        pipeline.scoring_engine.weights = request.weights

    res = pipeline.analyze_cv_and_job(
        cv_text=request.cv_text,
        job_text=request.job_description_text
    )

    return AnalysisResponse(
        structured_cv=res["structured_cv"],
        structured_job_description=res["structured_jd"],
        matching_result=res["matching_result"],
        semantic_similarity=res["semantic_similarity_raw"]
    )


@app.post("/api/analyze/file", response_model=AnalysisResponse)
async def analyze_file(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):
    if not job_description.strip():
        raise HTTPException(status_code=400, detail="job_description form field is required.")

    content = await file.read()
    ats_warnings = []
    cv_text = ""

    if file.filename.lower().endswith(".pdf"):
        parsed = PDFParser.extract_text_from_bytes(content)
        cv_text = parsed["text"]
        ats_warnings = parsed["ats_warnings"]
    else:
        # Plain text / markdown
        cv_text = content.decode("utf-8", errors="ignore")

    if not cv_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract readable text from uploaded file.")

    pipeline = get_pipeline()
    res = pipeline.analyze_cv_and_job(
        cv_text=cv_text,
        job_text=job_description,
        ats_warnings=ats_warnings
    )

    return AnalysisResponse(
        structured_cv=res["structured_cv"],
        structured_job_description=res["structured_jd"],
        matching_result=res["matching_result"],
        semantic_similarity=res["semantic_similarity_raw"]
    )
