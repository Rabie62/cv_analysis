"""Explainable Hybrid Scoring Engine.

Combines deterministic rule-based checks, taxonomy normalization,
and embedding similarity into transparent sub-scores (0-100) and an overall score.
"""

from typing import Dict, List, Set, Optional, Tuple
import re
from src.schemas import (
    StructuredCV,
    StructuredJobDescription,
    SubScores,
    MatchingResult,
    EvidencePoint,
    ActionableRecommendations,
    BulletImprovement,
)
from src.taxonomy import SkillNormalizer

DEFAULT_WEIGHTS = {
    "skills_match": 0.35,
    "experience_relevance": 0.25,
    "seniority_fit": 0.15,
    "education_fit": 0.10,
    "impact_and_achievements": 0.10,
    "ats_readiness": 0.05,
}


class ScoringEngine:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or DEFAULT_WEIGHTS
        # Ensure weights sum to 1.0
        total_w = sum(self.weights.values())
        self.weights = {k: v / total_w for k, v in self.weights.items()}

    def score(
        self,
        cv: StructuredCV,
        jd: StructuredJobDescription,
        raw_cv_text: str = "",
        semantic_sim: Optional[float] = None
    ) -> MatchingResult:
        strengths: List[EvidencePoint] = []
        gaps: List[EvidencePoint] = []
        recs = ActionableRecommendations()

        # 1. Skills Match Sub-score
        skills_score, matched_must, missing_must, matched_nice, missing_nice = (
            self._compute_skills_score(cv, jd)
        )
        recs.missing_critical_keywords = list(missing_must)

        if matched_must:
            strengths.append(EvidencePoint(
                statement=f"Matches {len(matched_must)} critical required skills: {', '.join(sorted(matched_must)[:5])}",
                cv_evidence_quote=", ".join(sorted(matched_must)[:4]),
                jd_reference="Must-Have Skills",
                is_verified_in_cv=True
            ))

        if missing_must:
            gaps.append(EvidencePoint(
                statement=f"Missing {len(missing_must)} essential must-have skills: {', '.join(sorted(missing_must))}",
                cv_evidence_quote=None,
                jd_reference="Must-Have Skills",
                is_verified_in_cv=True
            ))
            recs.skill_gap_guidance.append(
                f"Prioritize building or showcasing experience in: {', '.join(sorted(missing_must))}."
            )

        # 2. Seniority & Experience Fit Sub-score
        seniority_score = self._compute_seniority_score(cv, jd)
        req_years = jd.requirements.min_years_experience or 0.0
        cand_years = cv.total_years_experience or 0.0
        if cand_years >= req_years and req_years > 0:
            strengths.append(EvidencePoint(
                statement=f"Meets seniority requirement ({cand_years:.1f} years vs {req_years:.1f} requested)",
                cv_evidence_quote=f"Total experience: ~{cand_years:.1f} years",
                jd_reference=f"Requires {req_years}+ years experience",
                is_verified_in_cv=True
            ))
        elif req_years > 0 and cand_years < req_years:
            gaps.append(EvidencePoint(
                statement=f"Years of experience below target ({cand_years:.1f} years vs {req_years:.1f} requested)",
                cv_evidence_quote=f"Total experience: ~{cand_years:.1f} years",
                jd_reference=f"Requires {req_years}+ years experience",
                is_verified_in_cv=True
            ))

        # 3. Education & Certifications Fit Sub-score
        education_score = self._compute_education_score(cv, jd)

        # 4. Impact & Achievements Sub-score
        impact_score, weak_bullets = self._compute_impact_score(cv)
        for wb in weak_bullets[:3]:
            recs.bullet_rewrites.append(BulletImprovement(
                original_bullet=wb,
                suggested_rewrite=f"Spearheaded {wb.lower().replace('- ', '')}, optimizing performance and delivering measurable ROI.",
                rationale="Rewrite using the XYZ action-oriented formula: Accomplished [X] as measured by [Y] by doing [Z]."
            ))

        # 5. Experience Semantic Relevance (Embedding or lexical fallback)
        if semantic_sim is not None:
            # Scale cosine similarity (typically 0.25 - 0.75) to 10-100
            experience_score = max(10.0, min(100.0, ((semantic_sim - 0.20) / 0.55) * 100.0))
        else:
            # Fallback heuristic
            experience_score = (skills_score * 0.6) + (seniority_score * 0.4)

        # 6. ATS Readiness Sub-score
        ats_score, ats_advice = self._compute_ats_readiness(cv, raw_cv_text)
        recs.structural_advice.extend(ats_advice)

        sub_scores = SubScores(
            skills_match=round(skills_score, 1),
            experience_relevance=round(experience_score, 1),
            seniority_fit=round(seniority_score, 1),
            education_fit=round(education_score, 1),
            impact_and_achievements=round(impact_score, 1),
            ats_readiness=round(ats_score, 1)
        )

        overall = (
            sub_scores.skills_match * self.weights["skills_match"] +
            sub_scores.experience_relevance * self.weights["experience_relevance"] +
            sub_scores.seniority_fit * self.weights["seniority_fit"] +
            sub_scores.education_fit * self.weights["education_fit"] +
            sub_scores.impact_and_achievements * self.weights["impact_and_achievements"] +
            sub_scores.ats_readiness * self.weights["ats_readiness"]
        )

        return MatchingResult(
            overall_score=round(overall, 1),
            sub_scores=sub_scores,
            score_weights_applied=self.weights,
            strengths=strengths,
            gaps=gaps,
            recommendations=recs,
            hallucination_rate=0.0
        )

    def _compute_skills_score(
        self, cv: StructuredCV, jd: StructuredJobDescription
    ) -> Tuple[float, Set[str], Set[str], Set[str], Set[str]]:
        # Collect all candidate skills
        cv_raw = set(cv.raw_skills_list)
        for cat in cv.skills:
            cv_raw.update(cat.skills)
        for exp in cv.work_experience:
            cv_raw.update(exp.technologies_used)

        must_have = set(jd.requirements.must_have_skills)
        nice_have = set(jd.requirements.nice_to_have_skills)

        matched_must, missing_must = SkillNormalizer.match_skills(cv_raw, must_have)
        matched_nice, missing_nice = SkillNormalizer.match_skills(cv_raw, nice_have)

        must_cov = (len(matched_must) / len(must_have)) if must_have else 1.0
        nice_cov = (len(matched_nice) / len(nice_have)) if nice_have else 1.0

        score = (must_cov * 0.75 + nice_cov * 0.25) * 100.0
        return max(0.0, min(100.0, score)), matched_must, missing_must, matched_nice, missing_nice

    def _compute_seniority_score(self, cv: StructuredCV, jd: StructuredJobDescription) -> float:
        req = jd.requirements.min_years_experience
        if not req or req <= 0:
            return 85.0
        cand = cv.total_years_experience or 0.0
        if cand >= req:
            return 100.0
        return max(20.0, (cand / req) * 100.0)

    def _compute_education_score(self, cv: StructuredCV, jd: StructuredJobDescription) -> float:
        if not jd.requirements.required_degrees:
            return 85.0
        # Check if candidate has degree
        has_degree = len(cv.education) > 0
        if not has_degree:
            return 40.0
        return 90.0

    def _compute_impact_score(self, cv: StructuredCV) -> Tuple[float, List[str]]:
        total_bullets = 0
        quantified_count = 0
        weak_bullets = []

        metric_regex = re.compile(r"(\d+[\.,]?\d*[%kKmM\$€]|accrued|reduced|increased|improved|saved|\b\d+\b)")

        for exp in cv.work_experience:
            # Check explicit quantified achievements
            quantified_count += len(exp.quantified_achievements)
            for resp in exp.responsibilities:
                total_bullets += 1
                if metric_regex.search(resp):
                    quantified_count += 1
                else:
                    weak_bullets.append(resp)

        if total_bullets == 0:
            return 60.0, weak_bullets

        ratio = min(1.0, quantified_count / max(1, total_bullets))
        score = 40.0 + (ratio * 60.0)
        return round(score, 1), weak_bullets

    def _compute_ats_readiness(
        self, cv: StructuredCV, raw_text: str
    ) -> Tuple[float, List[str]]:
        score = 100.0
        advice = []

        if cv.ats_formatting_issues:
            penalty = len(cv.ats_formatting_issues) * 10
            score -= penalty
            advice.extend(cv.ats_formatting_issues)

        # Content length check
        word_count = len(raw_text.split()) if raw_text else 300
        if word_count < 150:
            score -= 20
            advice.append("CV is very short (<150 words). Provide more detailed responsibilities and metrics.")
        elif word_count > 1200:
            score -= 10
            advice.append("CV exceeds standard length (>1200 words). Aim for a concise 1-2 page format.")

        return max(30.0, score), advice
