# Privacy, Data Protection & GDPR Compliance Notice

## 1. Regulatory Context
Under the **General Data Protection Regulation (EU) 2016/679 (GDPR)**, resumes/CVs contain high-risk Personally Identifiable Information (PII) including real names, email addresses, phone numbers, home addresses, employment history, and occasionally sensitive demographic proxies.

This project implements **Privacy-by-Design and Privacy-by-Default (GDPR Art. 25)**.

---

## 2. Key Privacy Guarantees

### Zero Third-Party Data Transmission (Fully Local Execution)
- All PDF parsing, LLM inference, embedding generation, and scoring occur **strictly on the local host machine**.
- No candidate resumes or extracted text are transmitted to external commercial APIs (e.g. OpenAI, Anthropic, Google Cloud, or third-party cloud aggregators).
- Network requests are restricted solely to initial local model weight downloads.

### Automated PII Sanitization
- Prior to saving any CV into evaluation datasets or logging pipelines, raw files undergo automated redaction via `src/privacy/anonymizer.py`:
  - **Emails** are replaced with synthetic aliases (`candidate_xyz@redacted-applicant.org`).
  - **Phone numbers** are masked (`+XX-XXX-XXX-XXXX`).
  - **Social profiles (LinkedIn, GitHub)** are pseudonymized.
  - Candidate names are substituted with randomized synthetic identifiers.

### Data Minimization (Art. 5(1)(c))
- The scoring engine only extracts and retains features directly relevant to professional qualifications:
  - Technical skills
  - Seniority and work responsibilities
  - Educational degrees and certifications
- Contact information is excluded from all scoring weights and matching calculations.

---

## 3. Synthetic Data Policy
All test datasets, benchmark suites, and demonstration examples included in this repository (`eval/benchmark_data/`) are 100% synthetically generated or derived from public, permissible open-license datasets with all personal markers anonymized.
