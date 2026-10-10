"""Synthetic CV and Job Description Generation Engine.

Generates paired CVs and Job Descriptions across multiple technical domains
in French and English with 4 calibrated match tiers:
- Tier 1: Strong Match (85-98)
- Tier 2: Moderate Match (60-75)
- Tier 3: Career Pivot / Weak Match (35-50)
- Tier 4: Total Mismatch / Negative Control (<25)
"""
import sys
from pathlib import Path
from typing import Dict, List, Any
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anonymizer import PIIAnonymizer

DOMAINS = [
    {
        "id": "backend_swe_en",
        "language": "en",
        "title": "Senior Backend Engineer (Python/Distributed Systems)",
        "jd_text": """
Position: Senior Backend Engineer (Python/Distributed Systems)
Location: Remote / New York, NY
Experience Required: 5+ years

Role Overview:
We are seeking an experienced Backend Engineer to scale our transactional services. You will design high-throughput APIs, optimize database queries, and lead microservices architecture.

Must-Have Requirements:
- 5+ years of software development experience with Python (FastAPI or Django)
- Deep expertise in PostgreSQL, query optimization, and Redis caching
- Hands-on experience with Docker, Kubernetes, and AWS cloud infrastructure
- Track record of designing RESTful microservices handling high traffic
- Fluency in English

Nice-to-Have:
- Experience with Go or Rust
- Familiarity with Apache Kafka or RabbitMQ
- Experience mentoring junior developers
""",
        "must_have_skills": ["Python", "FastAPI", "PostgreSQL", "Redis", "Docker", "Kubernetes", "AWS"],
        "nice_to_have_skills": ["Go", "Kafka", "CI/CD"],
        "min_years": 5.0,
        "profiles": {
            "tier_1_strong": {
                "years_experience": 6.5,
                "education": "M.S. in Computer Science, Georgia Tech",
                "cv_text": """
Alex Morgan | email: alex.morgan@example.com | phone: +1-555-8392 | New York, NY
GitHub: github.com/alexmorgan | LinkedIn: linkedin.com/in/alexmorgan

Summary:
Senior Backend Engineer with 6.5 years specializing in high-throughput Python microservices, distributed caching, and cloud infrastructure on AWS.

Skills:
Languages: Python, Go, SQL, Bash
Frameworks & Databases: FastAPI, Django, PostgreSQL, Redis, RabbitMQ
DevOps & Cloud: AWS (ECS, RDS, S3), Docker, Kubernetes, GitHub Actions, Terraform

Experience:
Senior Software Engineer | FinStream Corp | New York, NY | 2021-02 - Present
- Architected transactional API gateway using FastAPI and Redis, reducing p99 latency by 44% across 20M daily calls.
- Scaled PostgreSQL database with connection pooling and query optimization, handling $4M+ in daily transaction volume.
- Led migration of 8 core services to Kubernetes clusters on AWS with zero downtime.

Backend Engineer | DataMesh Inc | Boston, MA | 2018-06 - 2021-01
- Developed distributed data ingestion services using Python and RabbitMQ, processing 1.2 TB of event streams daily.
- Automated CI/CD deployment pipelines using Docker and GitHub Actions, cutting deployment time from 45 min to 8 min.

Education:
M.S. in Computer Science | Georgia Institute of Technology | 2018
B.S. in Computer Engineering | Rutgers University | 2016
""",
                "ground_truth_scores": {
                    "skills_match": 95.0,
                    "experience_relevance": 95.0,
                    "seniority_fit": 100.0,
                    "education_fit": 95.0,
                    "impact_and_achievements": 92.0,
                    "ats_readiness": 95.0,
                    "overall_score": 95.5
                },
                "human_rubric": {"skills": 5, "experience": 5, "seniority": 5, "education": 5, "impact": 5, "ats": 5}
            },
            "tier_2_moderate": {
                "years_experience": 4.0,
                "education": "B.S. in Computer Science, State University",
                "cv_text": """
Jordan Taylor | email: jordan.taylor@example.com | phone: +1-555-4819 | Remote
LinkedIn: linkedin.com/in/jordantaylor

Summary:
Backend Python Developer with 4 years of experience building web applications, REST APIs, and working with relational databases.

Skills:
Python, Django, Flask, PostgreSQL, MySQL, Docker, AWS (basic), Git

Experience:
Backend Developer | AppNova Studios | 2021-03 - Present
- Built RESTful API endpoints for client dashboard using Django and PostgreSQL.
- Maintained relational database schemas and wrote migrations.
- Containerized development workflows using Docker.

Junior Python Developer | WebSphere | 2019-09 - 2021-02
- Implemented user authentication and account management services using Flask.
- Fixed bug tickets and wrote unit tests for core API features.

Education:
B.S. in Computer Science | State University | 2019
""",
                "ground_truth_scores": {
                    "skills_match": 68.0,
                    "experience_relevance": 70.0,
                    "seniority_fit": 80.0,
                    "education_fit": 85.0,
                    "impact_and_achievements": 55.0,
                    "ats_readiness": 85.0,
                    "overall_score": 71.0
                },
                "human_rubric": {"skills": 4, "experience": 3, "seniority": 4, "education": 4, "impact": 3, "ats": 4}
            },
            "tier_3_pivot": {
                "years_experience": 3.5,
                "education": "B.A. in Economics & Data Analysis",
                "cv_text": """
Casey Bennett | email: casey.b@example.com | phone: +1-555-1928 | Chicago, IL
LinkedIn: linkedin.com/in/caseybennett

Summary:
Data Analyst with 3.5 years of experience analyzing business metrics with Python and SQL. Looking to transition into backend software development.

Skills:
Python, SQL, Pandas, NumPy, Tableau, Excel, Basic Git

Experience:
Data Analyst | MarketPulse | 2021-01 - Present
- Wrote complex SQL queries on PostgreSQL data warehouse to generate executive financial reports.
- Automated data extraction scripts using Python and Pandas, saving 10 hours of manual work weekly.
- Built interactive Tableau dashboards for marketing campaigns.

Business Intelligence Assistant | RetailMetrics | 2019-08 - 2020-12
- Verified data accuracy in CRM and produced monthly KPI summaries.

Education:
B.A. in Economics | University of Illinois | 2019
""",
                "ground_truth_scores": {
                    "skills_match": 32.0,
                    "experience_relevance": 40.0,
                    "seniority_fit": 70.0,
                    "education_fit": 60.0,
                    "impact_and_achievements": 50.0,
                    "ats_readiness": 80.0,
                    "overall_score": 45.0
                },
                "human_rubric": {"skills": 2, "experience": 2, "seniority": 3, "education": 3, "impact": 2, "ats": 4}
            },
            "tier_4_mismatch": {
                "years_experience": 5.0,
                "education": "B.F.A. in Graphic Design",
                "cv_text": """
Riley Chen | email: riley.chen@example.com | phone: +1-555-7721 | Los Angeles, CA
Portfolio: rileychendesign.com

Summary:
Creative Graphic Designer & Art Director with 5 years creating brand identities, editorial layouts, and social media visuals.

Skills:
Adobe Creative Suite, Photoshop, Illustrator, InDesign, Figma, Typography, Brand Identity

Experience:
Senior Graphic Designer | Horizon Creative Studio | 2021-05 - Present
- Designed visual branding for over 30 retail and hospitality clients.
- Supervised digital marketing campaigns across Instagram and print media.

Graphic Designer | PrintCraft Media | 2018-09 - 2021-04
- Created brand guidelines, packaging designs, and brochure layouts.

Education:
B.F.A. in Graphic Design | California College of the Arts | 2018
""",
                "ground_truth_scores": {
                    "skills_match": 0.0,
                    "experience_relevance": 12.0,
                    "seniority_fit": 100.0,
                    "education_fit": 40.0,
                    "impact_and_achievements": 40.0,
                    "ats_readiness": 75.0,
                    "overall_score": 22.0
                },
                "human_rubric": {"skills": 1, "experience": 1, "seniority": 5, "education": 2, "impact": 2, "ats": 3}
            }
        }
    },
    {
        "id": "mlops_engineer_fr",
        "language": "fr",
        "title": "Ingénieur MLOps & Déploiement IA",
        "jd_text": """
Poste: Ingénieur MLOps & Déploiement IA
Lieu: Paris / Hybride
Expérience requise: 3+ ans d'expérience

Missions:
Au sein du pôle Data & IA, vous assurerez l'industrialisation des modèles de Machine Learning et NLP, de l'orchestration au monitoring en production.

Profil recherché (Must-Have):
- 3+ ans d'expérience en MLOps ou Data Engineering
- Excellente maîtrise de Python et des frameworks ML (PyTorch, Scikit-Learn)
- Expertise en conteneurisation et orchestration: Docker, Kubernetes, Airflow
- Expérience concrète sur une plateforme cloud (GCP ou AWS)
- Mise en place de pipelines CI/CD pour le ML (MLflow, Git)
- Maîtrise du français et de l'anglais technique

Atouts (Nice-to-Have):
- Connaissance de Hugging Face et des LLMs
- Optimisation d'inférence (ONNX, TensorRT)
- Expérience avec les bases de données vectorielles
""",
        "must_have_skills": ["Python", "PyTorch", "Docker", "Kubernetes", "Apache Airflow", "MLflow", "GCP"],
        "nice_to_have_skills": ["Hugging Face", "ONNX", "CI/CD"],
        "min_years": 3.0,
        "profiles": {
            "tier_1_strong": {
                "years_experience": 4.0,
                "education": "Diplôme d'Ingénieur en Informatique (Bac+5), Télécom Paris",
                "cv_text": """
Maxime Girard | Email: maxime.girard@example.fr | Tél: +33 6 44 22 11 00 | Paris, France
GitHub: github.com/maximegirard | LinkedIn: linkedin.com/in/maximegirard

Profil:
Ingénieur MLOps avec 4 ans d'expérience dans l'industrialisation, le déploiement sur Kubernetes et l'optimisation de modèles de Machine Learning sur GCP.

Compétences:
Langages: Python, Bash, SQL
ML & Data: PyTorch, Scikit-Learn, MLflow, Apache Airflow, Hugging Face
DevOps & Cloud: GCP (Vertex AI), Docker, Kubernetes, Helm, GitLab CI, DVC

Expériences Professionnelles:
Ingénieur MLOps | AI Solutions France | Paris | 2021-04 - Présent
- Déployé 14 modèles de NLP sur Kubernetes (GKE), réduisant la latence d'inférence de 45% grâce à ONNX Runtime.
- Automatisé les pipelines de réentraînement continus avec Apache Airflow et MLflow pour 350k requêtes journalières.
- Implémenté une architecture de monitoring de drift de données diminuant les incidents de 60%.

Data Engineer / MLOps | RetailData | Lyon | 2019-10 - 2021-03
- Conçu des pipelines ETL et conteneurisé les modèles de recommandation avec Docker et GCP.
- Mis en place les tests unitaires et d'intégration dans les pipelines GitLab CI.

Formation:
Diplôme d'Ingénieur en Informatique (Bac+5) | Télécom Paris | 2014 - 2019
""",
                "ground_truth_scores": {
                    "skills_match": 98.0,
                    "experience_relevance": 96.0,
                    "seniority_fit": 100.0,
                    "education_fit": 100.0,
                    "impact_and_achievements": 90.0,
                    "ats_readiness": 95.0,
                    "overall_score": 96.5
                },
                "human_rubric": {"skills": 5, "experience": 5, "seniority": 5, "education": 5, "impact": 5, "ats": 5}
            },
            "tier_2_moderate": {
                "years_experience": 2.5,
                "education": "Master Informatique (Bac+5), Université de Lille",
                "cv_text": """
Camille Rousseau | Email: camille.rousseau@example.fr | Tél: +33 6 99 88 77 66 | Lille, France
LinkedIn: linkedin.com/in/camillerousseau

Profil:
Data Scientist junior avec 2,5 ans d'expérience en modélisation Python et début de pratique MLOps (Docker, MLflow).

Compétences:
Python, Scikit-Learn, PyTorch, Pandas, Docker, MLflow, Git, SQL

Expériences:
Data Scientist | LogiTech | Lille | 2021-09 - Présent
- Développé des modèles prédictifs de séries temporelles avec Python et Scikit-Learn.
- Expérimenté le tracking de métriques avec MLflow et conteneurisé des scripts sous Docker.

Stage Data Science | FinGroup | 2021-02 - 2021-08
- Nettoyage de données et exploration statistique sous Pandas.

Formation:
Master Informatique et Données | Université de Lille | 2021
""",
                "ground_truth_scores": {
                    "skills_match": 65.0,
                    "experience_relevance": 65.0,
                    "seniority_fit": 83.0,
                    "education_fit": 90.0,
                    "impact_and_achievements": 50.0,
                    "ats_readiness": 85.0,
                    "overall_score": 68.5
                },
                "human_rubric": {"skills": 3, "experience": 3, "seniority": 4, "education": 5, "impact": 2, "ats": 4}
            },
            "tier_3_pivot": {
                "years_experience": 3.0,
                "education": "Licence Professionnelle Administration Réseaux",
                "cv_text": """
Lucas Moreau | Email: lucas.moreau@example.fr | Tél: +33 6 11 22 33 44 | Rennes, France
LinkedIn: linkedin.com/in/lucasmoreau

Profil:
Administrateur Systèmes & Linux avec 3 ans d'expérience sur l'infrastructure sur site et Docker. Souhaite s'orienter vers les technologies MLOps.

Compétences:
Linux, Bash, Docker, Ansible, Administration Réseaux, Notions de Python, Git

Expériences:
Administrateur Systèmes | Ouest Telecom | 2020-10 - Présent
- Gestion de serveurs Debian et déploiement de conteneurs Docker pour les outils internes.
- Automatisation des tâches de maintenance système avec des scripts Bash et Ansible.

Technicien Support | InfoRennes | 2019-09 - 2020-09
- Support utilisateurs et installation d'équipements réseaux.

Formation:
Licence Pro Réseaux et Systèmes | IUT Rennes | 2019
""",
                "ground_truth_scores": {
                    "skills_match": 35.0,
                    "experience_relevance": 45.0,
                    "seniority_fit": 100.0,
                    "education_fit": 60.0,
                    "impact_and_achievements": 45.0,
                    "ats_readiness": 80.0,
                    "overall_score": 48.0
                },
                "human_rubric": {"skills": 2, "experience": 2, "seniority": 5, "education": 3, "impact": 2, "ats": 4}
            },
            "tier_4_mismatch": {
                "years_experience": 4.0,
                "education": "Master en Droit des Affaires",
                "cv_text": """
Sophie Lefevre | Email: sophie.lefevre@example.fr | Tél: +33 6 77 66 55 44 | Paris, France
LinkedIn: linkedin.com/in/sophielefevre

Profil:
Juriste d'entreprise spécialisée en droit commercial et conformité des contrats, avec 4 ans d'expérience en cabinet et groupe bancaire.

Compétences:
Droit commercial, Rédaction de contrats, Conformité RGPD, Négociation juridique, Suite Office

Expériences:
Juriste Contrats | Banque Nationale | 2021-03 - Présent
- Rédaction et validation des contrats fournisseurs et accords de partenariat commercial.
- Audit de conformité des processus internes avec le règlement RGPD.

Juriste Junior | Cabinet Juridique Martin | 2019-06 - 2021-02
- Veille juridique et préparation des dossiers de contentieux commercial.

Formation:
Master 2 en Droit des Affaires | Université Paris 1 Panthéon-Sorbonne | 2019
""",
                "ground_truth_scores": {
                    "skills_match": 0.0,
                    "experience_relevance": 10.0,
                    "seniority_fit": 100.0,
                    "education_fit": 30.0,
                    "impact_and_achievements": 40.0,
                    "ats_readiness": 80.0,
                    "overall_score": 21.0
                },
                "human_rubric": {"skills": 1, "experience": 1, "seniority": 5, "education": 1, "impact": 2, "ats": 4}
            }
        }
    }
]


class SyntheticDatasetGenerator:
    """Generates synthetic CV-JD dataset with controlled match tiers and PII sanitization."""

    @staticmethod
    def generate_benchmark_dataset(target_total_pairs: int = 160) -> List[Dict[str, Any]]:
        dataset = []
        pair_id = 1

        # Calculate replicates needed to reach target_total_pairs
        base_pairs_count = len(DOMAINS) * 4  # 2 domains * 4 tiers = 8 base archetypes
        repeats = max(1, target_total_pairs // base_pairs_count)

        for rep in range(repeats):
            for domain in DOMAINS:
                jd_text = domain["jd_text"].strip()
                jd_id = f"{domain['id']}_rep{rep+1}"

                for tier_key, profile in domain["profiles"].items():
                    raw_cv = profile["cv_text"].strip()
                    cand_id = f"CAND_{domain['language'].upper()}_{pair_id:04d}"

                    # Anonymize through PII Sanitizer
                    sanitized_cv, _ = PIIAnonymizer.anonymize_text(raw_cv, candidate_id=cand_id)

                    sample_record = {
                        "pair_id": f"PAIR_{pair_id:04d}",
                        "language": domain["language"],
                        "domain": domain["id"],
                        "target_job_title": domain["title"],
                        "match_tier": tier_key,
                        "candidate_id": cand_id,
                        "raw_cv": sanitized_cv,
                        "raw_jd": jd_text,
                        "must_have_skills": domain["must_have_skills"],
                        "nice_to_have_skills": domain["nice_to_have_skills"],
                        "min_years_experience": domain["min_years"],
                        "candidate_years_experience": profile["years_experience"],
                        "candidate_degree": profile["education"],
                        "ground_truth_scores": profile["ground_truth_scores"],
                        "human_rubric_scores": profile["human_rubric"]
                    }
                    dataset.append(sample_record)
                    pair_id += 1

        return dataset
