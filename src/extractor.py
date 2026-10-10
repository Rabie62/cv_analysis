"""CV and Job Description Extraction Module.

Extracts structured data from raw CV text and Job Descriptions using local LLMs
with Pydantic validation and deterministic fallbacks.
"""

import json
import re
from typing import Optional, Dict, Any
import httpx

from src.schemas import StructuredCV, PersonalInfo, StructuredJobDescription, JobRequirements
from src.taxonomy import SkillNormalizer

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

CV_EXTRACTION_PROMPT = """You are an expert ATS (Applicant Tracking System) parser.
Extract structured information from the provided CV/Resume into valid JSON matching this schema:
{
  "language": "en" or "fr",
  "personal_info": {
    "full_name": string or null,
    "email": string or null,
    "phone": string or null,
    "location": string or null,
    "linkedin": string or null,
    "github": string or null
  },
  "summary": string or null,
  "total_years_experience": float or null,
  "skills": [
    {"category": "Programming Languages", "skills": ["..."]},
    {"category": "Frameworks", "skills": ["..."]}
  ],
  "raw_skills_list": ["skill1", "skill2"],
  "work_experience": [
    {
      "title": string,
      "company": string,
      "location": string or null,
      "start_date": string or null,
      "end_date": string or null,
      "responsibilities": ["..."],
      "quantified_achievements": ["..."],
      "technologies_used": ["..."],
      "evidence_span": string or null
    }
  ],
  "education": [
    {
      "degree": string,
      "field_of_study": string or null,
      "institution": string,
      "graduation_year": int or null
    }
  ],
  "certifications": [
    {"name": string, "issuer": string or null, "year": int or null}
  ],
  "languages": [
    {"language": string, "proficiency": string}
  ],
  "ats_formatting_issues": []
}

Output ONLY raw valid JSON. Do not write explanation or markdown."""

JD_EXTRACTION_PROMPT = """You are an expert HR recruiter.
Extract the requirements from the Job Description into valid JSON matching this schema:
{
  "title": string,
  "company": string or null,
  "location": string or null,
  "language": "en" or "fr",
  "summary": string or null,
  "responsibilities": ["..."],
  "requirements": {
    "min_years_experience": float or null,
    "seniority_level": string or null,
    "required_degrees": ["..."],
    "must_have_skills": ["..."],
    "nice_to_have_skills": ["..."],
    "required_certifications": ["..."],
    "required_languages": ["..."]
  }
}

Output ONLY raw valid JSON."""


class StructuredExtractor:
    def __init__(self, model_name: str = "qwen2.5:3b", ollama_url: str = OLLAMA_GENERATE_URL):
        self.model_name = model_name
        self.ollama_url = ollama_url

    def extract_cv(self, raw_text: str) -> StructuredCV:
        """Extracts structured CV information from raw text."""
        # 1. Fallback regex extractions for contact info
        detected_email = self._extract_email(raw_text)
        detected_phone = self._extract_phone(raw_text)

        payload = {
            "model": self.model_name,
            "prompt": f"CV Content:\n{raw_text}",
            "system": CV_EXTRACTION_PROMPT,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 2048}
        }

        try:
            resp = httpx.post(self.ollama_url, json=payload, timeout=90.0)
            resp.raise_for_status()
            data = resp.json().get("response", "{}")
            parsed = json.loads(data)
            structured = StructuredCV.model_validate(parsed)

            # Backfill contact info if LLM missed it
            if not structured.personal_info.email and detected_email:
                structured.personal_info.email = detected_email
            if not structured.personal_info.phone and detected_phone:
                structured.personal_info.phone = detected_phone

            # Normalize raw skills through taxonomy
            if structured.raw_skills_list:
                normalized = [SkillNormalizer.normalize_skill(s) for s in structured.raw_skills_list]
                structured.raw_skills_list = list(dict.fromkeys(normalized))

            return structured

        except Exception as e:
            # Fallback heuristic parser if LLM is unavailable or fails
            return self._heuristic_cv_fallback(raw_text, detected_email, detected_phone)

    def extract_job_description(self, raw_text: str) -> StructuredJobDescription:
        """Extracts structured Job Description from raw text."""
        payload = {
            "model": self.model_name,
            "prompt": f"Job Description:\n{raw_text}",
            "system": JD_EXTRACTION_PROMPT,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 2048}
        }

        try:
            resp = httpx.post(self.ollama_url, json=payload, timeout=90.0)
            resp.raise_for_status()
            data = resp.json().get("response", "{}")
            parsed = json.loads(data)
            return StructuredJobDescription.model_validate(parsed)
        except Exception:
            return self._heuristic_jd_fallback(raw_text)

    @staticmethod
    def _extract_email(text: str) -> Optional[str]:
        m = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
        return m.group(0) if m else None

    @staticmethod
    def _extract_phone(text: str) -> Optional[str]:
        m = re.search(r"(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}", text)
        return m.group(0) if m else None

    def _heuristic_cv_fallback(self, raw_text: str, email: Optional[str], phone: Optional[str]) -> StructuredCV:
        return StructuredCV(
            language="fr" if any(w in raw_text.lower() for w in ["expérience", "formation", "compétences"]) else "en",
            personal_info=PersonalInfo(email=email, phone=phone),
            summary=raw_text[:200] if len(raw_text) > 200 else raw_text,
            raw_skills_list=[]
        )

    def _heuristic_jd_fallback(self, raw_text: str) -> StructuredJobDescription:
        return StructuredJobDescription(
            title="Extracted Job Position",
            language="fr" if any(w in raw_text.lower() for w in ["poste", "missions", "profil"]) else "en",
            summary=raw_text[:200]
        )
