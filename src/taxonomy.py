"""Skill Taxonomy Normalization and Canonicalization.

Maps aliases, abbreviations, and localized terms (FR/EN) to canonical concepts
aligned with standardized skill frameworks (e.g. ESCO).
"""

from typing import Dict, Set, Optional, Tuple

# Canonical alias mapping: alias.lower() -> canonical_name
SKILL_ALIASES: Dict[str, str] = {
    # Programming Languages
    "python": "Python",
    "py": "Python",
    "python3": "Python",
    "javascript": "JavaScript",
    "js": "JavaScript",
    "ecmascript": "JavaScript",
    "typescript": "TypeScript",
    "ts": "TypeScript",
    "golang": "Go",
    "go": "Go",
    "c++": "C++",
    "cpp": "C++",
    "c#": "C#",
    "csharp": "C#",
    "java": "Java",
    "rust": "Rust",
    "sql": "SQL",
    "pl/sql": "PL/SQL",

    # Web & Backend Frameworks
    "fastapi": "FastAPI",
    "django": "Django",
    "flask": "Flask",
    "react": "React",
    "react.js": "React",
    "reactjs": "React",
    "vue": "Vue.js",
    "vue.js": "Vue.js",
    "vuejs": "Vue.js",
    "angular": "Angular",
    "angularjs": "Angular",
    "node": "Node.js",
    "node.js": "Node.js",
    "nodejs": "Node.js",
    "express": "Express.js",
    "spring": "Spring Boot",
    "spring boot": "Spring Boot",
    "next.js": "Next.js",
    "nextjs": "Next.js",

    # Data, ML & AI
    "pytorch": "PyTorch",
    "torch": "PyTorch",
    "tensorflow": "TensorFlow",
    "tf": "TensorFlow",
    "keras": "Keras",
    "scikit-learn": "Scikit-Learn",
    "sklearn": "Scikit-Learn",
    "pandas": "Pandas",
    "polars": "Polars",
    "numpy": "NumPy",
    "hugging face": "Hugging Face",
    "transformers": "Transformers",
    "mlflow": "MLflow",
    "airflow": "Apache Airflow",
    "apache airflow": "Apache Airflow",
    "dvc": "DVC",
    "spark": "Apache Spark",
    "apache spark": "Apache Spark",
    "pyspark": "PySpark",

    # Cloud, DevOps & Infrastructure
    "docker": "Docker",
    "docker container": "Docker",
    "k8s": "Kubernetes",
    "kubernetes": "Kubernetes",
    "aws": "AWS",
    "amazon web services": "AWS",
    "gcp": "GCP",
    "google cloud": "GCP",
    "google cloud platform": "GCP",
    "azure": "Microsoft Azure",
    "microsoft azure": "Microsoft Azure",
    "ci/cd": "CI/CD",
    "cicd": "CI/CD",
    "continuous integration": "CI/CD",
    "github actions": "GitHub Actions",
    "gitlab ci": "GitLab CI",
    "jenkins": "Jenkins",
    "terraform": "Terraform",
    "ansible": "Ansible",
    "linux": "Linux",
    "bash": "Bash/Shell",
    "shell": "Bash/Shell",
    "git": "Git",

    # Databases & Caching
    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "psql": "PostgreSQL",
    "mysql": "MySQL",
    "redis": "Redis",
    "mongodb": "MongoDB",
    "mongo": "MongoDB",
    "elasticsearch": "Elasticsearch",
    "elastic search": "Elasticsearch",
    "dynamodb": "DynamoDB",
    "sqlite": "SQLite"
}


class SkillNormalizer:
    """Normalizes raw skill mentions to canonical standard skill names."""

    @staticmethod
    def normalize_skill(skill: str) -> str:
        clean = skill.strip().lower()
        # Direct dictionary match
        if clean in SKILL_ALIASES:
            return SKILL_ALIASES[clean]
        
        # Strip common prefixes/suffixes like "(proficient)", "(intermediate)", etc.
        for noise in ["(advanced)", "(intermediate)", "(beginner)", "(notions)", "- senior"]:
            if noise in clean:
                clean = clean.replace(noise, "").strip()
                if clean in SKILL_ALIASES:
                    return SKILL_ALIASES[clean]

        # Return title-cased if unmapped
        return skill.strip()

    @staticmethod
    def normalize_skill_set(skills: Set[str]) -> Set[str]:
        return {SkillNormalizer.normalize_skill(s) for s in skills if s.strip()}

    @staticmethod
    def match_skills(cv_skills: Set[str], jd_skills: Set[str]) -> Tuple[Set[str], Set[str]]:
        """Given CV skills and JD skills, returns (matched_skills, missing_skills)."""
        norm_cv = SkillNormalizer.normalize_skill_set(cv_skills)
        norm_jd = SkillNormalizer.normalize_skill_set(jd_skills)

        # Matched set is intersection
        matched = norm_cv.intersection(norm_jd)
        missing = norm_jd.difference(norm_cv)
        return matched, missing
