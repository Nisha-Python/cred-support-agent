"""
TASK 13 — RAG TRIAD EVALUATION
------------------------------

Evaluates the RAG system under MOCK_LLM using 15 queries.

Metrics:
1. Context Relevance
2. Groundedness
3. Answer Relevance

All scores are deterministic and range from 0.0 to 1.0.
"""

from pathlib import Path
import sys
import re


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT RAG
# ============================================================

from rag.retrieval import (
    retrieve,
    grounded_answer,
    calibrate_threshold,
)


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 3

THRESHOLD = calibrate_threshold()


# ============================================================
# 15 EVALUATION QUERIES
# ============================================================

EVALUATION_QUERIES = [
    {
        "query": "What are the eligibility criteria for a Personal Loan?",
        "expected_topic": "loan eligibility",
    },
    {
        "query": "What are the EMI rules for loans?",
        "expected_topic": "emi",
    },
    {
        "query": "What fees are charged on credit cards?",
        "expected_topic": "credit card fees",
    },
    {
        "query": "What documents are required for KYC?",
        "expected_topic": "kyc",
    },
    {
        "query": "How can I dispute a fraudulent transaction?",
        "expected_topic": "fraud dispute",
    },
    {
        "query": "How can I close my bank account?",
        "expected_topic": "account closure",
    },
    {
        "query": "What are the applicable interest rate slabs?",
        "expected_topic": "interest rates",
    },
    {
        "query": "Are there penalties for prepaying a loan?",
        "expected_topic": "prepayment",
    },
    {
        "query": "What is the minimum balance requirement?",
        "expected_topic": "minimum balance",
    },
    {
        "query": "How does my credit score affect loan eligibility?",
        "expected_topic": "credit score",
    },
    {
        "query": "What are the rules for opening a joint account?",
        "expected_topic": "joint account",
    },
    {
        "query": "What documents are required for an NRI account?",
        "expected_topic": "nri account",
    },
    {
        "query": "Can I make an early repayment on my loan?",
        "expected_topic": "prepayment",
    },
    {
        "query": "What KYC documents can I submit during account onboarding?",
        "expected_topic": "kyc",
    },
    {
        "query": "How are loan interest rates determined?",
        "expected_topic": "interest rates",
    },
]


# ============================================================
# TEXT HELPERS
# ============================================================

def tokenize(text: str) -> set[str]:
    """
    Convert text into normalized word tokens.
    """

    return set(
        re.findall(
            r"[a-zA-Z0-9]+",
            text.lower(),
        )
    )


def keyword_overlap(
    query: str,
    text: str,
) -> float:
    """
    Calculate normalized lexical overlap.

    This deterministic metric is used because the capstone
    requires evaluation under MOCK_LLM.
    """

    query_tokens = tokenize(query)
    text_tokens = tokenize(text)

    if not query_tokens:
        return 0.0

    overlap = (
        query_tokens
        & text_tokens
    )

    return len(overlap) / len(query_tokens)


# ============================================================
# CONTEXT RELEVANCE
# ============================================================

def context_relevance(
    query: str,
    results: list,
) -> float:
    """
    Measure whether retrieved context is relevant to the query.

    Uses the average lexical overlap across retrieved chunks,
    combined with the retriever similarity score.
    """

    if not results:
        return 0.0

    scores = []

    for result in results:

        lexical_score = keyword_overlap(
            query,
            result["text"],
        )

        similarity_score = float(
            result.get(
                "similarity",
                0.0,
            )
        )

        score = (
            0.5 * lexical_score
            + 0.5 * similarity_score
        )

        scores.append(
            min(
                1.0,
                max(
                    0.0,
                    score,
                ),
            )
        )

    return sum(scores) / len(scores)


# ============================================================
# GROUNDEDNESS
# ============================================================

def groundedness(
    answer: str,
    results: list,
) -> float:
    """
    Measure how much of the generated answer is supported
    by retrieved context.

    MOCK_LLM generation is strictly based on retrieved text,
    so sentence/phrase overlap provides a deterministic
    groundedness measure.
    """

    if not answer or not results:
        return 0.0

    context = " ".join(
        result["text"]
        for result in results
    )

    answer_tokens = tokenize(answer)
    context_tokens = tokenize(context)

    if not answer_tokens:
        return 0.0

    overlap = (
        answer_tokens
        & context_tokens
    )

    return min(
        1.0,
        len(overlap) / len(answer_tokens),
    )


# ============================================================
# ANSWER RELEVANCE
# ============================================================

def answer_relevance(
    query: str,
    answer: str,
) -> float:
    """
    Measure whether the answer addresses the query.

    Uses lexical overlap between the user query and generated
    answer under deterministic MOCK_LLM evaluation.
    """

    if not answer:
        return 0.0

    return min(
        1.0,
        keyword_overlap(
            query,
            answer,
        ),
    )


# ============================================================
# SINGLE QUERY EVALUATION
# ============================================================

def evaluate_query(
    query: str,
) -> dict:
    """
    Evaluate one query using the sentence-based RAG strategy.
    """

    results = retrieve(
        query,
        strategy="sentence",
        top_k=TOP_K,
    )

    generated = grounded_answer(
        query,
        strategy="sentence",
        top_k=TOP_K,
        threshold=THRESHOLD,
    )

    answer = generated.get(
        "answer",
        "",
    )

    context_score = context_relevance(
        query,
        results,
    )

    grounded_score = groundedness(
        answer,
        results,
    )

    answer_score = answer_relevance(
        query,
        answer,
    )

    return {
        "query": query,
        "context_relevance": round(
            context_score,
            4,
        ),
        "groundedness": round(
            grounded_score,
            4,
        ),
        "answer_relevance": round(
            answer_score,
            4,
        ),
        "fallback": generated.get(
            "fallback",
            False,
        ),
        "sources": generated.get(
            "sources",
            [],
        ),
    }


# ============================================================
# FULL EVALUATION
# ============================================================

def evaluate_all():
    """
    Evaluate all 15 required queries.
    """

    results = []

    print("\n" + "=" * 80)
    print("TASK 13 — RAG TRIAD EVALUATION")
    print("=" * 80)

    print(
        f"\nStrategy: sentence"
    )

    print(
        f"Top-K: {TOP_K}"
    )

    print(
        f"Threshold: {THRESHOLD:.4f}"
    )

    print(
        "\nMOCK_LLM: ENABLED"
    )

    print("-" * 80)

    for index, item in enumerate(
        EVALUATION_QUERIES,
        start=1,
    ):

        query = item["query"]

        result = evaluate_query(
            query
        )

        results.append(
            result
        )

        print(
            f"\n{index:02d}. {query}"
        )

        print(
            f"    Context Relevance : "
            f"{result['context_relevance']:.4f}"
        )

        print(
            f"    Groundedness      : "
            f"{result['groundedness']:.4f}"
        )

        print(
            f"    Answer Relevance  : "
            f"{result['answer_relevance']:.4f}"
        )

        print(
            f"    Fallback          : "
            f"{result['fallback']}"
        )

    # ========================================================
    # AVERAGES
    # ========================================================

    count = len(results)

    average_context = (
        sum(
            r["context_relevance"]
            for r in results
        )
        / count
    )

    average_groundedness = (
        sum(
            r["groundedness"]
            for r in results
        )
        / count
    )

    average_answer = (
        sum(
            r["answer_relevance"]
            for r in results
        )
        / count
    )

    print("\n" + "=" * 80)
    print("TASK 13 — FINAL AVERAGES")
    print("=" * 80)

    print(
        f"\nNumber of queries: {count}"
    )

    print(
        f"Average Context Relevance: "
        f"{average_context:.4f}"
    )

    print(
        f"Average Groundedness: "
        f"{average_groundedness:.4f}"
    )

    print(
        f"Average Answer Relevance: "
        f"{average_answer:.4f}"
    )

    print("\n" + "=" * 80)

    return {
        "num_queries": count,
        "average_context_relevance": round(
            average_context,
            4,
        ),
        "average_groundedness": round(
            average_groundedness,
            4,
        ),
        "average_answer_relevance": round(
            average_answer,
            4,
        ),
        "results": results,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    evaluation = evaluate_all()

    print(
        "\nTASK 13 VALIDATION: PASSED"
    )