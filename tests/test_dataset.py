"""Unit tests for Phase 2 dataset generation, anonymization, and splits."""

import json
from pathlib import Path
import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_dataset_files_exist():
    full_path = DATA_DIR / "annotated" / "dataset.json"
    train_path = DATA_DIR / "splits" / "train.json"
    val_path = DATA_DIR / "splits" / "val.json"
    test_path = DATA_DIR / "splits" / "test.json"

    assert full_path.exists(), "annotated/dataset.json must exist"
    assert train_path.exists(), "splits/train.json must exist"
    assert val_path.exists(), "splits/val.json must exist"
    assert test_path.exists(), "splits/test.json must exist"


def test_dataset_splits_no_leakage():
    with open(DATA_DIR / "splits" / "train.json", "r", encoding="utf-8") as f:
        train = json.load(f)
    with open(DATA_DIR / "splits" / "val.json", "r", encoding="utf-8") as f:
        val = json.load(f)
    with open(DATA_DIR / "splits" / "test.json", "r", encoding="utf-8") as f:
        test = json.load(f)

    train_ids = set(x["pair_id"] for x in train)
    val_ids = set(x["pair_id"] for x in val)
    test_ids = set(x["pair_id"] for x in test)

    # Zero overlap between splits
    assert len(train_ids.intersection(val_ids)) == 0, "Train and Val must not overlap"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test must not overlap"
    assert len(val_ids.intersection(test_ids)) == 0, "Val and Test must not overlap"

    # Total check
    assert len(train) + len(val) + len(test) == 160


def test_anonymization_and_pii_redaction():
    with open(DATA_DIR / "annotated" / "dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    for record in data:
        raw_cv = record["raw_cv"]
        # Must not contain unmasked phone numbers like +1-555 or +33 6
        assert "+1-555" not in raw_cv, f"Unmasked phone found in {record['pair_id']}"
        assert "+33 6" not in raw_cv, f"Unmasked phone found in {record['pair_id']}"
        # Emails must be masked to @redacted-applicant.org
        assert "@redacted-applicant.org" in raw_cv, f"Masked email missing in {record['pair_id']}"


def test_tier_score_monotonicity():
    with open(DATA_DIR / "annotated" / "dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Group scores by tier
    tier_scores = {}
    for record in data:
        tier = record["match_tier"]
        score = record["ground_truth_scores"]["overall_score"]
        tier_scores.setdefault(tier, []).append(score)

    avg_tier1 = sum(tier_scores["tier_1_strong"]) / len(tier_scores["tier_1_strong"])
    avg_tier2 = sum(tier_scores["tier_2_moderate"]) / len(tier_scores["tier_2_moderate"])
    avg_tier3 = sum(tier_scores["tier_3_pivot"]) / len(tier_scores["tier_3_pivot"])
    avg_tier4 = sum(tier_scores["tier_4_mismatch"]) / len(tier_scores["tier_4_mismatch"])

    # Strict monotonicity: Tier 1 > Tier 2 > Tier 3 > Tier 4
    assert avg_tier1 > avg_tier2 > avg_tier3 > avg_tier4
    assert avg_tier1 >= 85.0
    assert 60.0 <= avg_tier2 <= 75.0
    assert 35.0 <= avg_tier3 <= 50.0
    assert avg_tier4 <= 25.0
