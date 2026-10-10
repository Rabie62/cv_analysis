"""End-to-end CV Analysis Pipeline.

Orchestrates:
1. Document ingestion (PDF or text)
2. Layout parsing & ATS diagnostics
3. Structured extraction (via local LLM)
4. Semantic embedding similarity computation
5. Deterministic scoring & gap analysis
6. Evidence grounding verification (hallucination rejection)
"""

from typing import Dict, Any, Optional
from sentence_transformers import SentenceTransformer, util

from src.pdf_parser import PDFParser
from src.schemas import StructuredCV, StructuredJobDescription, MatchingResult
from src.extractor import StructuredExtractor
from src.scoring_engine import ScoringEngine


class CVAnalysisPipeline:
    def __init__(
        self,
        llm_model: str = "qwen2.5:3b",
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        scoring_weights: Optional[Dict[str, float]] = None
    ):
        self.extractor = StructuredExtractor(model_name=llm_model)
        self.embedding_model = SentenceTransformer(embedding_model_name, device="cpu")
        self.scoring_engine = ScoringEngine(weights=scoring_weights)

    def analyze_cv_and_job(
        self,
        cv_text: str,
        job_text: str,
        ats_warnings: Optional[list] = None
    ) -> Dict[str, Any]:
        """Runs the complete analysis pipeline on raw CV text and Job Description text."""
        # 1. Structured Extraction
        structured_cv = self.extractor.extract_cv(cv_text)
        if ats_warnings:
            structured_cv.ats_formatting_issues.extend(ats_warnings)

        structured_jd = self.extractor.extract_job_description(job_text)

        # 2. Semantic Embedding Similarity between CV experience and JD responsibilities
        cv_work_text = " ".join([
            f"{exp.title} at {exp.company}: " + " ".join(exp.responsibilities)
            for exp in structured_cv.work_experience
        ])
        cv_summary_repr = f"{structured_cv.summary or ''} {cv_work_text} {' '.join(structured_cv.raw_skills_list)}".strip()
        
        jd_resp_text = " ".join(structured_jd.responsibilities)
        jd_reqs_repr = f"{structured_jd.title}. {structured_jd.summary or ''}. {jd_resp_text}. {' '.join(structured_jd.requirements.must_have_skills)}".strip()

        emb_cv = self.embedding_model.encode(cv_summary_repr, convert_to_tensor=True)
        emb_jd = self.embedding_model.encode(jd_reqs_repr, convert_to_tensor=True)
        semantic_sim = float(util.cos_sim(emb_cv, emb_jd).item())

        # 3. Hybrid Scoring
        match_result: MatchingResult = self.scoring_engine.score(
            cv=structured_cv,
            jd=structured_jd,
            raw_cv_text=cv_text,
            semantic_sim=semantic_sim
        )

        # 4. Evidence Grounding Verification Step
        verified_result = self._verify_evidence_grounding(match_result, cv_text)

        return {
            "structured_cv": structured_cv.model_dump(),
            "structured_jd": structured_jd.model_dump(),
            "matching_result": verified_result.model_dump(),
            "semantic_similarity_raw": round(semantic_sim, 3)
        }

    def analyze_pdf_file(self, pdf_path: str, job_text: str) -> Dict[str, Any]:
        """Parses a PDF file and runs the full analysis pipeline."""
        parsed_doc = PDFParser.extract_text_from_file(pdf_path)
        return self.analyze_cv_and_job(
            cv_text=parsed_doc["text"],
            job_text=job_text,
            ats_warnings=parsed_doc["ats_warnings"]
        )

    @staticmethod
    def _verify_evidence_grounding(result: MatchingResult, raw_cv_text: str) -> MatchingResult:
        """Rejects or flags any strength claim whose cited quote does not appear in the CV."""
        raw_lower = raw_cv_text.lower()
        unverified_count = 0
        total_claims = len(result.strengths)

        for strength in result.strengths:
            if strength.cv_evidence_quote:
                # Check substring presence
                quote_clean = strength.cv_evidence_quote.strip().lower()
                # Split multi-item quotes (e.g., skill lists)
                quote_tokens = [q.strip() for q in quote_clean.split(",") if q.strip()]
                found = any(tok in raw_lower for tok in quote_tokens)
                if not found:
                    strength.is_verified_in_cv = False
                    unverified_count += 1
                else:
                    strength.is_verified_in_cv = True

        if total_claims > 0:
            result.hallucination_rate = round((unverified_count / total_claims) * 100.0, 1)
        else:
            result.hallucination_rate = 0.0

        return result
