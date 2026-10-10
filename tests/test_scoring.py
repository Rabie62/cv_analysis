"""Unit tests for the hybrid scoring engine and skill taxonomy normalizer."""

import pytest
from src.schemas import StructuredCV, WorkExperience, Education, StructuredJobDescription, JobRequirements
from src.scoring_engine import ScoringEngine
from src.taxonomy import SkillNormalizer


def test_skill_normalizer():
    assert SkillNormalizer.normalize_skill("JS") == "JavaScript"
    assert SkillNormalizer.normalize_skill("python3") == "Python"
    assert SkillNormalizer.normalize_skill("k8s") == "Kubernetes"
    assert SkillNormalizer.normalize_skill("Postgres") == "PostgreSQL"

    cv_skills = {"js", "python", "docker", "gcp"}
    jd_skills = {"javascript", "python3", "kubernetes"}
    matched, missing = SkillNormalizer.match_skills(cv_skills, jd_skills)

    assert "JavaScript" in matched
    assert "Python" in matched
    assert "Kubernetes" in missing


def test_scoring_engine_high_match():
    cv = StructuredCV(
        language="en",
        total_years_experience=6.0,
        raw_skills_list=["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"],
        work_experience=[
            WorkExperience(
                title="Senior Backend Engineer",
                company="CloudTech",
                responsibilities=["Built microservices improving response time by 40% across 5M requests"],
                quantified_achievements=["Reduced latency by 40%"]
            )
        ],
        education=[
            Education(degree="B.S. Computer Science", institution="MIT", graduation_year=2018)
        ]
    )

    jd = StructuredJobDescription(
        title="Senior Python Backend Developer",
        requirements=JobRequirements(
            min_years_experience=5.0,
            must_have_skills=["Python", "FastAPI", "PostgreSQL"],
            nice_to_have_skills=["AWS", "Docker"]
        )
    )

    engine = ScoringEngine()
    result = engine.score(cv, jd)

    assert result.overall_score >= 80.0
    assert result.sub_scores.skills_match == 100.0
    assert result.sub_scores.seniority_fit == 100.0
    assert len(result.strengths) > 0
    assert len(result.gaps) == 0


def test_scoring_engine_gap_detection():
    # Junior CV applying for Senior role with missing skills
    cv = StructuredCV(
        language="en",
        total_years_experience=1.0,
        raw_skills_list=["HTML", "CSS"],
        work_experience=[
            WorkExperience(
                title="Junior Web Assistant",
                company="Agency",
                responsibilities=["Updated web page contents"]
            )
        ]
    )

    jd = StructuredJobDescription(
        title="Senior Python Backend Developer",
        requirements=JobRequirements(
            min_years_experience=5.0,
            must_have_skills=["Python", "FastAPI", "PostgreSQL"],
            nice_to_have_skills=["AWS"]
        )
    )

    engine = ScoringEngine()
    result = engine.score(cv, jd)

    assert result.overall_score < 50.0
    assert result.sub_scores.skills_match == 0.0
    assert len(result.gaps) >= 2
    assert "Python" in result.recommendations.missing_critical_keywords
