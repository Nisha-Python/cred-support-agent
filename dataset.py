import random
from collections import Counter

# ============================================================
# Cred Support Agent - Loan Application Dataset
# ============================================================

SEED = 20260906
NUM_RECORDS = 50

# Dataset design choices
CATEGORIES = [
    "Personal Loan",
    "Home Loan",
    "Auto Loan",
    "Education Loan",
    "Business Loan",
]

STATUSES = [
    "Submitted",
    "Under Review",
    "Approved",
    "Rejected",
    "Disbursed",
]

# Realistic loan range for a student-built banking dataset.
MIN_LOAN_AMOUNT = 50_000
MAX_LOAN_AMOUNT = 5_000_000

random.seed(SEED)

LOAN_APPLICATIONS = []

for i in range(1, NUM_RECORDS + 1):
    record = {
        "record_id": f"CRD-{i:04d}",
        "category": random.choice(CATEGORIES),
        "status": random.choice(STATUSES),
        "loan_amount_inr": random.randint(
            MIN_LOAN_AMOUNT,
            MAX_LOAN_AMOUNT
        ),
        "days_since_created": random.randint(0, 30),
        "flagged_for_fraud_review": random.random() < 0.20,
    }

    LOAN_APPLICATIONS.append(record)


# ============================================================
# Validation
# ============================================================

def validate_dataset():
    assert len(LOAN_APPLICATIONS) >= 40

    # Required categories
    category_counts = Counter(
        record["category"] for record in LOAN_APPLICATIONS
    )

    for category in CATEGORIES:
        assert category_counts[category] >= 3, (
            f"Category '{category}' has fewer than 3 records."
        )

    # Required statuses
    status_counts = Counter(
        record["status"] for record in LOAN_APPLICATIONS
    )

    for status in STATUSES:
        assert status_counts[status] >= 1, (
            f"Status '{status}' is missing."
        )

    # Validate individual records
    for record in LOAN_APPLICATIONS:
        assert isinstance(record["record_id"], str)
        assert record["category"] in CATEGORIES
        assert record["status"] in STATUSES
        assert MIN_LOAN_AMOUNT <= record["loan_amount_inr"] <= MAX_LOAN_AMOUNT
        assert 0 <= record["days_since_created"] <= 30
        assert isinstance(record["flagged_for_fraud_review"], bool)

    # Fraud percentage
    fraud_count = sum(
        record["flagged_for_fraud_review"]
        for record in LOAN_APPLICATIONS
    )

    fraud_percentage = (fraud_count / NUM_RECORDS) * 100

    assert 10 <= fraud_percentage <= 30, (
        f"Fraud percentage {fraud_percentage:.2f}% "
        "is outside the required 10%-30% range."
    )

    return category_counts, status_counts, fraud_percentage


# ============================================================
# Run validation and report results
# ============================================================

if __name__ == "__main__":
    category_counts, status_counts, fraud_percentage = validate_dataset()

    print("=" * 60)
    print("CRED LOAN APPLICATION DATASET")
    print("=" * 60)

    print(f"\nTotal records: {len(LOAN_APPLICATIONS)}")
    print(f"Random seed: {SEED}")
    print(
        f"Loan amount range: ₹{MIN_LOAN_AMOUNT:,} - "
        f"₹{MAX_LOAN_AMOUNT:,}"
    )

    print("\nCategory counts:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")

    print("\nStatus counts:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    print(
        f"\nFraud-review records: "
        f"{sum(r['flagged_for_fraud_review'] for r in LOAN_APPLICATIONS)}"
    )
    print(f"Fraud-review percentage: {fraud_percentage:.2f}%")

    print("\nValidation: PASSED")
    print("=" * 60)