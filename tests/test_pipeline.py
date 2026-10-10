"""Unit tests for the end-to-end pipeline and evidence grounding."""

import pytest
from unittest.mock import MagicMock
from src.pipeline import CVAnalysisPipeline
from src.schemas import (
    StructuredCV,
    WorkExperience,
    StructuredJobDescription,
    JobRequirements,
    MatchingResult,
    EvidencePoint,
    SubScores,
    ActionableRecommendations,
)


def test_evidence_grounding_verification():
    raw_cv = "Experienced Python developer with 5 years in FastAPI, PostgreSQL, and Docker containerization."

    # Test case 1: Real quote present in CV text
    true_strength = EvidencePoint(
        statement="Matches required Python skill",
        cv_evidence_quote="FastAPI, PostgreSQL",
        is_verified_in_cv=True
    )

    # Test case 2: Hallucinated quote not present in CV text
    hallucinated_strength = EvidencePoint(
        statement="Expert in Kubernetes orchestration",
        cv_evidence_quote="Kubernetes clusters",
        is_verified_in_cv=True
    )

    result = MatchingResult(
        overall_score=85.0,
        sub_scores=SubScores(
            skills_match=85.0,
            experience_relevance=80.0,
            seniority_fit=100.0,
            education_fit=80.0,
            impact_and_achievements=70.0,
            ats_readiness=90.0
        ),
        strengths=[true_strength, hallucinated_strength],
        recommendations=ActionableRecommendations()
    )

    verified = CVAnalysisPipeline._verify_evidence_grounding(result, raw_cv)

    # First claim should be verified
    assert verified.strengths[0].is_verified_in_cv is True
    # Second claim should fail verification
    assert verified.strengths[1].is_verified_in_cv is False
    # Hallucination rate: 1 out of 2 claims = 50.0%
    assert verified.hallucination_rate == 50.0
