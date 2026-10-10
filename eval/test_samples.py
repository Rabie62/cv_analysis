"""Curated multilingual (EN/FR) test samples for model selection and benchmark evaluation."""

TEST_SAMPLES = [
    {
        "id": "sample_en_swe",
        "language": "en",
        "description": "English Senior Software Engineer CV",
        "raw_cv": """
John Doe
Email: john.doe@example.com | Phone: +1-555-0199 | Location: San Francisco, CA
LinkedIn: linkedin.com/in/johndoe | GitHub: github.com/johndoe

Summary:
Full-Stack Senior Software Engineer with 6 years of experience building high-throughput distributed microservices, scalable REST APIs, and modern React interfaces.

Skills:
- Languages: Python, Go, TypeScript, SQL
- Frameworks & Tools: FastAPI, Django, React, PostgreSQL, Redis, Docker, Kubernetes, AWS, Git
- Methodologies: Agile/Scrum, CI/CD, Test-Driven Development (TDD)

Experience:
Senior Software Engineer | CloudScale Inc. | San Francisco, CA | 2021-03 - Present
- Architected and migrated monolithic backend to FastAPI microservices, reducing p99 API latency by 42% across 15M daily requests.
- Designed distributed Redis caching layer saving $18,000/month in PostgreSQL RDS infrastructure costs.
- Led and mentored a team of 4 junior and mid-level engineers, enforcing unit testing standards (raised coverage from 55% to 88%).

Software Engineer | FinTech Systems | Austin, TX | 2018-06 - 2021-02
- Developed secure payment processing pipelines using Go and PostgreSQL, handling over $2.5M in daily transactions.
- Automated deployment workflows with GitHub Actions and Docker, reducing release cycle time from 2 hours to 15 minutes.
- Implemented OAuth2 / JWT authentication service with role-based access control.

Education:
B.S. in Computer Science | University of Texas at Austin | 2014 - 2018
Honors: Magna Cum Laude, GPA 3.85/4.0

Certifications:
- AWS Certified Solutions Architect - Associate (2022)

Languages:
- English (Native)
- French (Basic / A2)
""",
        "ground_truth_skills": [
            "Python", "Go", "TypeScript", "SQL", "FastAPI", "Django", "React",
            "PostgreSQL", "Redis", "Docker", "Kubernetes", "AWS", "Git", "CI/CD"
        ],
        "years_experience": 6.0,
        "education_degree": "B.S. in Computer Science"
    },
    {
        "id": "sample_fr_datascience",
        "language": "fr",
        "description": "French Data Scientist / ML Engineer CV",
        "raw_cv": """
Amélie Laurent
Email: amelie.laurent@example.fr | Téléphone: +33 6 12 34 56 78 | Paris, France
LinkedIn: linkedin.com/in/amelielaurent | GitHub: github.com/amelielaurent

Profil:
Data Scientist & Ingénieure Machine Learning avec 3 ans d'expérience dans la conception de modèles de NLP et de vision par ordinateur, du prototypage à la mise en production sur GCP.

Compétences:
- Langages & Librairies: Python, SQL, PyTorch, Scikit-Learn, Hugging Face, Pandas, Polars
- MLOps & Cloud: GCP (Vertex AI), Docker, MLflow, Airflow, FastAPI, DVC, Git
- Langues: Français (Langue maternelle), Anglais (Courant / C1)

Expérience Professionnelle:
Data Scientist / ML Engineer | RetailAI | Paris | Mars 2022 - Présent
- Développé et déployé un pipeline NLP de classification de tickets clients avec RoBERTa, augmentant le taux de résolution automatique de 35% à 78%.
- Réduit le temps d'inférence des modèles de 60% via quantification ONNX Runtime et conteneurisation Docker.
- Mis en place le suivi des expérimentations avec MLflow et orchestré les pipelines de réentraînement hebdomadaires via Apache Airflow.

Stagiaire Data Science (PFE) | MedTech Analytics | Lyon | Septembre 2021 - Février 2022
- Entraîné des modèles de segmentation d'images IRM avec PyTorch et U-Net, atteignant un score Dice de 0.91.
- Rédigé la documentation technique et présenté les résultats aux équipes médicales.

Formation:
Diplôme d'Ingénieur en Informatique et Intelligence Artificielle (Bac+5) | INSA Lyon | 2017 - 2022
Mention: Très Bien

Certifications:
- Google Cloud Certified Associate Cloud Engineer (2023)
""",
        "ground_truth_skills": [
            "Python", "SQL", "PyTorch", "Scikit-Learn", "Hugging Face", "Pandas", "Polars",
            "GCP", "Vertex AI", "Docker", "MLflow", "Airflow", "FastAPI", "DVC", "Git"
        ],
        "years_experience": 3.0,
        "education_degree": "Diplôme d'Ingénieur en Informatique et Intelligence Artificielle (Bac+5)"
    }
]

TEST_JDS = [
    {
        "id": "jd_senior_python_en",
        "language": "en",
        "title": "Senior Backend Engineer (Python / Cloud)",
        "raw_text": """
Position: Senior Backend Engineer (Python / Cloud)
Location: Remote / Paris or San Francisco
Experience: 5+ years

About the Role:
We are looking for a Senior Backend Engineer to scale our distributed systems. You will lead architecture decisions, optimize high-throughput APIs, and mentor junior developers.

Must-Have Requirements:
- 5+ years of software development experience with Python (FastAPI or Django)
- Strong experience with relational databases (PostgreSQL) and caching (Redis)
- Proven experience with Docker, Kubernetes, and AWS or GCP
- Experience building scalable RESTful microservices with low latency
- Fluency in English

Nice-to-Have:
- Experience with Go or Rust
- Knowledge of CI/CD pipelines (GitHub Actions)
- Mentorship or tech lead experience
"""
    },
    {
        "id": "jd_mlops_fr",
        "language": "fr",
        "title": "Ingénieur Machine Learning & MLOps",
        "raw_text": """
Poste: Ingénieur Machine Learning & MLOps
Lieu: Paris / Hybride
Expérience: 2 à 4 ans d'expérience

Missions:
Au sein de l'équipe Data & IA, vous serez en charge de l'industrialisation, du déploiement et du monitoring de nos modèles de Machine Learning et NLP.

Profil recherché (Must-have):
- 2+ ans d'expérience en Data Science / ML Engineering
- Maîtrise de Python, PyTorch ou Scikit-Learn
- Solide expertise MLOps: Docker, MLflow, Airflow, CI/CD
- Expérience concrète sur un cloud public (GCP ou AWS)
- Diplôme Bac+5 (École d'ingénieurs ou équivalent universitaire)

Atouts (Nice-to-have):
- Expérience avec Hugging Face, Transformers et LLM
- Optimisation d'inférence (ONNX, TensorRT)
- Bonne communication en français et en anglais
"""
    }
]
