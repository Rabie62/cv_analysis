"""Pydantic data models for CV extraction, Job Descriptions, and Matching Results."""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


# =====================================================================
# 1. Candidate CV Models
# =====================================================================

class PersonalInfo(BaseModel):
    full_name: Optional[str] = Field(None, description="Candidate's full name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    location: Optional[str] = Field(None, description="City, Country or current location")
    linkedin: Optional[str] = Field(None, description="LinkedIn profile URL or handle")
    github: Optional[str] = Field(None, description="GitHub profile URL or handle")
    website: Optional[str] = Field(None, description="Personal website or portfolio URL")


class WorkExperience(BaseModel):
    title: str = Field(..., description="Job title / role")
    company: str = Field(..., description="Company or organization name")
    location: Optional[str] = Field(None, description="Location of work")
    start_date: Optional[str] = Field(None, description="Start date (e.g., '2021-06' or '2021')")
    end_date: Optional[str] = Field(None, description="End date or 'Present'")
    duration_months: Optional[int] = Field(None, description="Estimated duration in months")
    responsibilities: List[str] = Field(
        default_factory=list, description="Key responsibilities and duties"
    )
    quantified_achievements: List[str] = Field(
        default_factory=list,
        description="Measurable achievements with metrics (%, $, time saved, users reached)",
    )
    technologies_used: List[str] = Field(
        default_factory=list, description="Tools, frameworks, and technologies used in this role"
    )
    evidence_span: Optional[str] = Field(
        None, description="Exact or near-exact quote from the CV supporting this entry"
    )


class Education(BaseModel):
    degree: str = Field(..., description="Degree type (BSc, MSc, PhD, Engineering Diploma, etc.)")
    field_of_study: Optional[str] = Field(None, description="Field of study / major")
    institution: str = Field(..., description="University, school, or institution name")
    graduation_year: Optional[int] = Field(None, description="Year of graduation or expected")
    gpa_or_honors: Optional[str] = Field(None, description="Distinctions, GPA, or honors")


class Certification(BaseModel):
    name: str = Field(..., description="Certification name (e.g., AWS Solutions Architect)")
    issuer: Optional[str] = Field(None, description="Issuing body (e.g., AWS, Coursera, Scrum Alliance)")
    year: Optional[int] = Field(None, description="Year obtained")


class LanguageProficiency(BaseModel):
    language: str = Field(..., description="Language (e.g., French, English)")
    proficiency: str = Field(
        ...,
        description="Level of proficiency (Native, Fluent/C1-C2, Intermediate/B1-B2, Basic/A1-A2)",
    )


class SkillCategory(BaseModel):
    category: str = Field(
        ...,
        description="Category name (e.g., 'Programming Languages', 'Frameworks', 'DevOps & Cloud')",
    )
    skills: List[str] = Field(..., description="List of specific skills")


class StructuredCV(BaseModel):
    language: str = Field(
        "en", description="Detected primary language of the CV ('en' or 'fr')"
    )
    personal_info: PersonalInfo = Field(default_factory=PersonalInfo)
    summary: Optional[str] = Field(None, description="Executive summary or objective statement")
    total_years_experience: Optional[float] = Field(
        None, description="Total estimated years of relevant professional experience"
    )
    skills: List[SkillCategory] = Field(
        default_factory=list, description="Categorized list of skills"
    )
    raw_skills_list: List[str] = Field(
        default_factory=list, description="Flat list of all normalized individual skills"
    )
    work_experience: List[WorkExperience] = Field(
        default_factory=list, description="Chronological work history"
    )
    education: List[Education] = Field(
        default_factory=list, description="Educational background"
    )
    certifications: List[Certification] = Field(
        default_factory=list, description="Certifications and licenses"
    )
    languages: List[LanguageProficiency] = Field(
        default_factory=list, description="Language competencies"
    )
    ats_formatting_issues: List[str] = Field(
        default_factory=list,
        description="Potential ATS formatting issues detected (tables, multi-column, missing headers)",
    )


# =====================================================================
# 2. Job Description Models
# =====================================================================

class JobRequirements(BaseModel):
    min_years_experience: Optional[float] = Field(
        None, description="Minimum years of experience required"
    )
    seniority_level: Optional[str] = Field(
        None,
        description="Target seniority: 'Intern', 'Junior', 'Mid', 'Senior', 'Lead', 'Staff', 'Director'",
    )
    required_degrees: List[str] = Field(
        default_factory=list,
        description="Required or preferred degrees (e.g., Master's in Computer Science)",
    )
    must_have_skills: List[str] = Field(
        default_factory=list,
        description="Essential / non-negotiable hard and technical skills",
    )
    nice_to_have_skills: List[str] = Field(
        default_factory=list,
        description="Bonus / preferred skills that are not strictly mandatory",
    )
    required_certifications: List[str] = Field(
        default_factory=list,
        description="Required or preferred professional certifications",
    )
    required_languages: List[str] = Field(
        default_factory=list,
        description="Required languages (e.g., 'French Fluent', 'English Professional')",
    )


class StructuredJobDescription(BaseModel):
    title: str = Field(..., description="Job title / position name")
    company: Optional[str] = Field(None, description="Hiring company name")
    location: Optional[str] = Field(None, description="Job location or 'Remote / Hybride'")
    language: str = Field("en", description="Language of the job description ('en' or 'fr')")
    summary: Optional[str] = Field(None, description="Brief overview of the role and mission")
    responsibilities: List[str] = Field(
        default_factory=list, description="Primary duties and responsibilities"
    )
    requirements: JobRequirements = Field(default_factory=JobRequirements)


# =====================================================================
# 3. Matching Result & Evaluation Models
# =====================================================================

class EvidencePoint(BaseModel):
    statement: str = Field(..., description="The factual finding or observation")
    cv_evidence_quote: Optional[str] = Field(
        None, description="Exact or near-exact supporting quote extracted from the candidate's CV"
    )
    jd_reference: Optional[str] = Field(
        None, description="Corresponding requirement or section from the Job Description"
    )
    is_verified_in_cv: bool = Field(
        True, description="Whether the claim was verified against raw CV text"
    )


class SubScores(BaseModel):
    skills_match: float = Field(
        ..., ge=0.0, le=100.0, description="Coverage of must-have & nice-to-have skills (0-100)"
    )
    experience_relevance: float = Field(
        ..., ge=0.0, le=100.0, description="Semantic and domain relevance of previous work (0-100)"
    )
    seniority_fit: float = Field(
        ..., ge=0.0, le=100.0, description="Alignment between candidate years/level and JD requirements (0-100)"
    )
    education_fit: float = Field(
        ..., ge=0.0, le=100.0, description="Match on degree, field of study, and certifications (0-100)"
    )
    impact_and_achievements: float = Field(
        ..., ge=0.0, le=100.0, description="Presence of quantified results and business impact (0-100)"
    )
    ats_readiness: float = Field(
        ..., ge=0.0, le=100.0, description="Structural readability, ATS-friendliness, format clarity (0-100)"
    )


class BulletImprovement(BaseModel):
    original_bullet: str = Field(..., description="Weak or unquantified bullet from the CV")
    suggested_rewrite: str = Field(
        ..., description="Rewritten version using the XYZ format (Accomplished [X] by doing [Y], resulting in [Z])"
    )
    rationale: str = Field(..., description="Why this rewrite increases impact and JD relevance")


class ActionableRecommendations(BaseModel):
    missing_critical_keywords: List[str] = Field(
        default_factory=list, description="Must-have terms from the JD absent in the CV"
    )
    bullet_rewrites: List[BulletImprovement] = Field(
        default_factory=list, description="Concrete suggestions to improve CV bullet points"
    )
    structural_advice: List[str] = Field(
        default_factory=list, description="Formatting, layout, or section organization recommendations"
    )
    skill_gap_guidance: List[str] = Field(
        default_factory=list, description="Suggestions on how to address missing skills (e.g. projects, certs)"
    )


class MatchingResult(BaseModel):
    overall_score: float = Field(
        ..., ge=0.0, le=100.0, description="Weighted composite score between 0 and 100"
    )
    sub_scores: SubScores = Field(..., description="Breakdown of individual score dimensions")
    score_weights_applied: Dict[str, float] = Field(
        default_factory=dict, description="Weights used to calculate overall score"
    )
    strengths: List[EvidencePoint] = Field(
        default_factory=list, description="Evidence-backed candidate strengths matching the JD"
    )
    gaps: List[EvidencePoint] = Field(
        default_factory=list, description="Identified candidate weaknesses or missing requirements"
    )
    recommendations: ActionableRecommendations = Field(
        default_factory=ActionableRecommendations, description="Concrete, actionable improvements"
    )
    hallucination_rate: Optional[float] = Field(
        0.0, description="Percentage of claims rejected during evidence grounding verification"
    )
