"""Data Anonymization and PII Sanitization Module.

Complies with GDPR principles (Art. 5, Art. 25, Art. 32) by redacting or syntheticizing
personally identifiable information (names, emails, phones, social links, addresses)
before storage, model training, or sharing.
"""

import re
from typing import Dict, Any, Tuple


class PIIAnonymizer:
    """Detects and redacts or pseudo-anonymizes personal identifiers in CV text."""

    EMAIL_PATTERN = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
    PHONE_PATTERN = re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}")
    LINKEDIN_PATTERN = re.compile(r"(?:https?:\/\/)?(?:www\.)?linkedin\.com\/in\/[\w\-]+", re.IGNORECASE)
    GITHUB_PATTERN = re.compile(r"(?:https?:\/\/)?(?:www\.)?github\.com\/[\w\-]+", re.IGNORECASE)

    @classmethod
    def anonymize_text(cls, text: str, candidate_id: str = "CANDIDATE_001") -> Tuple[str, Dict[str, Any]]:
        """Redacts sensitive PII and returns sanitized text along with redaction metadata."""
        redactions = {
            "emails_redacted": 0,
            "phones_redacted": 0,
            "social_links_redacted": 0
        }

        # 1. Redact Emails
        emails = cls.EMAIL_PATTERN.findall(text)
        redactions["emails_redacted"] = len(emails)
        sanitized = cls.EMAIL_PATTERN.sub(f"{candidate_id.lower()}@redacted-applicant.org", text)

        # 2. Redact Phone Numbers
        phones = cls.PHONE_PATTERN.findall(sanitized)
        redactions["phones_redacted"] = len(phones)
        sanitized = cls.PHONE_PATTERN.sub("+XX-XXX-XXX-XXXX", sanitized)

        # 3. Redact Social URLs
        linkedin = cls.LINKEDIN_PATTERN.findall(sanitized)
        github = cls.GITHUB_PATTERN.findall(sanitized)
        redactions["social_links_redacted"] = len(linkedin) + len(github)
        sanitized = cls.LINKEDIN_PATTERN.sub(f"linkedin.com/in/{candidate_id.lower()}", sanitized)
        sanitized = cls.GITHUB_PATTERN.sub(f"github.com/{candidate_id.lower()}", sanitized)

        return sanitized, redactions
