"""Unit tests for Pydantic schema validation."""

import pytest
from src.schemas import (
    StructuredCV,
    WorkExperience,
    Education,
    StructuredJobDescription,
    JobRequirements,
    MatchingResult,
    SubScores,
    ActionableRecommendations,
)


def test_structured_cv_validation():
    cv = StructuredCV(
        language="en",
        personal_info={"full_name": "Alice Smith", "email": "alice@example.com"},
        total_years_experience=4.5,
        raw_skills_list=["Python", "FastAPI", "Docker"],
        work_experience=[
            WorkExperience(
                title="Software Engineer",
                company="TechCorp",
                responsibilities=["Developed REST APIs"],
                technologies_used=["Python", "FastAPI"]
            )
        ],
        education=[
            Education(
                degree="Bachelor of Science",
                institution="State University",
                graduation_year=2020
            )
        ]
    )
    assert cv.personal_info.full_name == "Alice Smith"
    assert cv.total_years_experience == 4.5
    assert len(cv.raw_skills_list) == 3


def test_matching_result_bounds():
    sub_scores = SubScores(
        skills_match=85.0,
        experience_relevance=90.0,
        seniority_fit=100.0,
        education_fit=80.0,
        impact_and_achievements=75.0,
        ats_readiness=95.0
    )
    res = MatchingResult(
        overall_score=87.5,
        sub_scores=sub_scores,
        recommendations=ActionableRecommendations()
    )
    assert 0.0 <= res.overall_score <= 100.0
