# CV-Job Matching Annotation & Scoring Guidelines

## 1. Overview
This guideline outlines the standardized rubric and scoring criteria used to create and evaluate the annotated CV/JD benchmark dataset. It guarantees consistent evaluation across human annotators and automated pipelines.

---

## 2. Multi-Criteria 1–5 Scoring Rubric

Each candidate CV is evaluated against the target Job Description on 6 orthogonal dimensions using an integer scale from **1 (Critically Deficient)** to **5 (Exceptional / Perfect Fit)**:

### 1. Skills Coverage (`skills_match`)
* **1 (Unacceptable):** No overlap with must-have technologies (<15% coverage).
* **2 (Minimal):** 15%–40% coverage. Possesses basic generic tools but lacks core technical stack.
* **3 (Partial / Acceptable):** 40%–65% coverage. Meets ~half of must-have requirements; lacks key libraries or secondary tools.
* **4 (Strong):** 70%–89% coverage. Meets all must-have skills, missing only 1 or 2 nice-to-haves.
* **5 (Exceptional):** 90%–100% coverage across both must-have and nice-to-have technical skills.

### 2. Experience Relevance (`experience_relevance`)
* **1:** Prior work history is in an entirely unrelated domain (e.g. Graphic Designer applying for Distributed Backend Engineer).
* **2:** Adjacent field with minimal transferable technical overlap (e.g. Business Analyst transitioning to Systems Programmer).
* **3:** Relevant general software experience, but different architectural scale or application domain.
* **4:** Directly relevant experience within similar industry verticals and comparable systems.
* **5:** Exact target domain fit (e.g., high-throughput transactional services, distributed cloud architectures).

### 3. Seniority & Career Progression (`seniority_fit`)
* **1:** Candidate has $<25\%$ of required years of experience (e.g. Intern applying for Senior).
* **2:** Significant shortfall (25%–50% of required years).
* **3:** Slight shortfall (50%–75% of required years) but demonstrates rapid learning.
* **4:** Meets requested experience (75%–100% of required years).
* **5:** Meets or exceeds requested experience with demonstrated technical leadership or team mentorship.

### 4. Education & Qualifications (`education_fit`)
* **1:** No relevant educational background or certifications.
* **2:** Non-technical degree with no specialized software certifications.
* **3:** Technical diploma or associate degree (e.g. Bac+2/3) for roles requiring Master's/Bac+5.
* **4:** Meets required degree (B.S. or Master's in Computer Science/Engineering).
* **5:** Top-tier engineering diploma (*Grandes Écoles*, M.S./Ph.D. in CS) plus certified cloud credentials (AWS Solutions Architect, GCP).

### 5. Impact & Quantified Achievements (`impact_and_achievements`)
* **1:** Pure task-based descriptions without any metrics or business context (*"Responsible for writing code"*).
* **2:** Primarily passive descriptions with occasional qualitative outcomes (*"Improved application speed"*).
* **3:** Mixed descriptions with at least 1 quantified achievement (*"Reduced latency by 20%"*).
* **4:** Strong action verbs and multiple quantified outcomes across several roles (*"Saved $18k/month", "Cut build time by 60%"*).
* **5:** Outstanding Google XYZ framework formatting (*"Accomplished [X] as measured by [Y] by doing [Z]"*) throughout entire resume.

### 6. ATS Readiness & Formatting Quality (`ats_readiness`)
* **1:** Severe ATS traps: un-OCRed scanned images, heavy embedded graphic icons, or corrupt text layers.
* **2:** Dense nested tables and multi-column sidebars that scramble reading order.
* **3:** Acceptable structure but formatting inconsistencies (varying bullet styles, missing headers).
* **4:** Standard clean layout with well-defined section headings (Experience, Education, Skills).
* **5:** Perfectly parsed semantic hierarchy, standard margins, clean contact header, and standard bullet formatting.

---

## 3. Ground Truth Composite Score Mapping

The overall score $Y \in [0, 100]$ maps to the 4 match tiers:

| Match Tier | Score Range | Expected Profile Profile |
| :--- | :---: | :--- |
| **Tier 1 (Strong Match)** | **85 – 100** | Exceeds must-haves, exact seniority, quantified impact, target degree. |
| **Tier 2 (Moderate Match)** | **60 – 75** | Meets core tools, missing 1–2 nice-to-haves, slightly lower seniority. |
| **Tier 3 (Career Pivot / Weak)** | **35 – 50** | Transferable soft skills, missing core frameworks, significant skill gap. |
| **Tier 4 (Total Mismatch)** | **0 – 25** | Unrelated discipline (negative control). |

---

## 4. Anonymization & Data Hygiene
All records must pass through `PIIAnonymizer`:
- Candidate names $\rightarrow$ `CAND_{LANG}_{ID}`
- Real emails $\rightarrow$ `@redacted-applicant.org`
- Real phones $\rightarrow$ `+XX-XXX-XXX-XXXX`
- Social handles $\rightarrow$ anonymized tokens
