from pathlib import Path
import os

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# ============================================================
# Environment
# ============================================================

load_dotenv(override=True)


# ============================================================
# MOCK_LLM CONFIGURATION
# ============================================================

# The capstone requires the graded system to work under
# deterministic MOCK_LLM without an API key or network access.
#
# Default is TRUE so the project works immediately after setup.
#
# To use a real LLM optionally:
#     MOCK_LLM=false
#
MOCK_LLM = os.getenv(
    "MOCK_LLM",
    "true"
).strip().lower() == "true"


# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

CHROMA_DIR = PROJECT_DIR / "chroma_db"

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# SentenceTransformer Embedding Model
# ============================================================

model = SentenceTransformer(
    MODEL_NAME
)


# ============================================================
# ChromaDB Persistent Client
# ============================================================

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)


# ============================================================
# Collections
# ============================================================

fixed_collection = client.get_collection(
    name="loan_kb_fixed"
)

sentence_collection = client.get_collection(
    name="loan_kb_sentence"
)


# ============================================================
# Similarity Helper
# ============================================================

def cosine_similarity_from_distance(
    distance
):
    """
    Convert Chroma cosine distance to cosine similarity.

    For normalized embeddings:

        similarity = 1 - cosine_distance
    """

    return 1.0 - distance


# ============================================================
# Retrieval
# ============================================================

def retrieve(
    query: str,
    strategy: str = "sentence",
    top_k: int = 3,
):
    """
    Retrieve the most relevant knowledge-base chunks.

    Args:
        query:
            User question.

        strategy:
            'sentence' or 'fixed'.

        top_k:
            Number of chunks to retrieve.

    Returns:
        list[dict]:
            Retrieved chunks containing:
            - text
            - doc_id
            - title
            - source
            - chunk_index
            - similarity
    """

    if not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    if strategy == "sentence":

        collection = sentence_collection

    elif strategy == "fixed":

        collection = fixed_collection

    else:

        raise ValueError(
            "strategy must be 'sentence' or 'fixed'"
        )


    # ========================================================
    # Query Embedding
    # ========================================================

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )[0]


    # ========================================================
    # Chroma Retrieval
    # ========================================================

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )


    # ========================================================
    # Convert Results
    # ========================================================

    retrieved = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):

        similarity = (
            cosine_similarity_from_distance(
                distance
            )
        )

        retrieved.append(
            {
                "text": document,
                "doc_id": metadata["doc_id"],
                "title": metadata["title"],
                "source": metadata["source"],
                "chunk_index": metadata["chunk_index"],
                "similarity": similarity,
            }
        )


    return retrieved


# ============================================================
# Threshold Calibration Queries
# ============================================================

IN_SCOPE_QUERIES = [

    "What are the eligibility criteria for a Personal Loan?",

    "How is EMI calculated for a loan?",

    "What documents are needed for KYC?",
]


OUT_OF_SCOPE_QUERIES = [

    "What is the weather forecast for tomorrow?",

    "Who won the latest cricket match?",
]


# ============================================================
# Threshold Calibration
# ============================================================

def calibrate_threshold():
    """
    Empirically calibrate the RAG similarity threshold.

    The calibration uses:

    - at least 3 in-scope queries
    - at least 2 out-of-scope queries

    The selected threshold is placed between:

        highest observed out-of-scope score

    and

        lowest observed in-scope score

    whenever the two observed clusters do not overlap.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "SIMILARITY THRESHOLD CALIBRATION"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # Score Containers
    # ========================================================

    in_scope_scores = []

    out_scope_scores = []


    # ========================================================
    # IN-SCOPE CALIBRATION
    # ========================================================

    print(
        "\nIN-SCOPE QUERIES"
    )

    print(
        "-" * 70
    )


    for query in IN_SCOPE_QUERIES:

        results = retrieve(
            query,
            strategy="sentence",
            top_k=3,
        )


        if not results:

            raise RuntimeError(
                "No retrieval results returned "
                f"for calibration query: {query}"
            )


        score = results[0]["similarity"]


        in_scope_scores.append(
            score
        )


        print(
            f"\nQuery: {query}"
        )

        print(
            f"Top-1 similarity: {score:.4f}"
        )

        print(
            f"Top document: "
            f"{results[0]['title']}"
        )


    # ========================================================
    # OUT-OF-SCOPE CALIBRATION
    # ========================================================

    print(
        "\n\nOUT-OF-SCOPE QUERIES"
    )

    print(
        "-" * 70
    )


    for query in OUT_OF_SCOPE_QUERIES:

        results = retrieve(
            query,
            strategy="sentence",
            top_k=3,
        )


        if not results:

            raise RuntimeError(
                "No retrieval results returned "
                f"for calibration query: {query}"
            )


        score = results[0]["similarity"]


        out_scope_scores.append(
            score
        )


        print(
            f"\nQuery: {query}"
        )

        print(
            f"Top-1 similarity: {score:.4f}"
        )

        print(
            f"Top document: "
            f"{results[0]['title']}"
        )


    # ========================================================
    # Calibration Summary
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CALIBRATION SUMMARY"
    )

    print(
        "=" * 70
    )


    print(
        "\nIn-scope scores:",
        [
            round(
                score,
                4
            )
            for score in in_scope_scores
        ]
    )


    print(
        "Out-of-scope scores:",
        [
            round(
                score,
                4
            )
            for score in out_scope_scores
        ]
    )


    # ========================================================
    # Observed Cluster Boundaries
    # ========================================================

    min_in_scope = min(
        in_scope_scores
    )

    max_out_scope = max(
        out_scope_scores
    )


    print(
        f"\nMinimum in-scope similarity: "
        f"{min_in_scope:.4f}"
    )


    print(
        f"Maximum out-of-scope similarity: "
        f"{max_out_scope:.4f}"
    )


    # ========================================================
    # Select Threshold
    # ========================================================

    if min_in_scope > max_out_scope:

        threshold = (
            min_in_scope
            + max_out_scope
        ) / 2


        print(
            f"\nChosen threshold: "
            f"{threshold:.4f}"
        )


        print(
            "Reason: threshold is midway between "
            "the highest observed out-of-scope score "
            "and the lowest observed in-scope score."
        )


    else:

        # If clusters overlap, use the lowest
        # in-scope score as a conservative threshold.

        threshold = min_in_scope


        print(
            "\nWARNING: observed similarity clusters overlap."
        )


        print(
            "Using the minimum in-scope score "
            "as a conservative threshold."
        )


        print(
            "Additional calibration queries are "
            "recommended before claiming strong separation."
        )


    return threshold


# ============================================================
# Optional Real LLM
# ============================================================

def get_llm():
    """
    Create the optional OpenAI chat model.

    This function is used ONLY when:

        MOCK_LLM=false

    The API key is loaded from the .env file.
    """

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )


    if not api_key:

        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Add your OpenAI API key to the .env file "
            "or use MOCK_LLM=true."
        )


    return ChatOpenAI(
        model="gpt-5.6-luna",
        api_key=api_key,
    )


# ============================================================
# Deterministic MOCK_LLM Generator
# ============================================================

def mock_llm_answer(
    query: str,
    results: list,
):
    """
    Deterministic local answer generator.

    This function intentionally does NOT call an external
    model or network service.

    It generates the answer only from retrieved KB text,
    which keeps the graded MOCK_LLM path deterministic.
    """

    if not results:

        return (
            "I don't know based on the available "
            "knowledge base."
        )


    # ========================================================
    # Use Retrieved Context Only
    # ========================================================

    answer_parts = []


    for result in results:

        text = (
            result.get(
                "text",
                ""
            )
            .strip()
        )


        if text:

            answer_parts.append(
                text
            )


    if not answer_parts:

        return (
            "I don't know based on the available "
            "knowledge base."
        )


    # ========================================================
    # Deterministic Answer
    # ========================================================

    return (
        "Based on the available knowledge base:\n\n"
        + "\n\n".join(
            answer_parts
        )
    )


# ============================================================
# Grounded Answer
# ============================================================

def grounded_answer(
    query: str,
    strategy: str = "sentence",
    top_k: int = 3,
    threshold: float | None = None,
):
    """
    Generate a grounded answer from the knowledge base.

    Process:

    1. Retrieve top-k chunks.
    2. Require an empirically calibrated threshold.
    3. Compare top-1 similarity with the threshold.
    4. Return the required fallback when similarity
       is below the threshold.
    5. Otherwise generate an answer using MOCK_LLM
       or an optional real LLM.
    6. Return sources and similarity.
    """

    # ========================================================
    # Validate Threshold
    # ========================================================

    if threshold is None:

        raise ValueError(
            "A calibrated threshold must be provided."
        )


    # ========================================================
    # Validate Threshold Value
    # ========================================================

    threshold = float(
        threshold
    )


    # ========================================================
    # Retrieve
    # ========================================================

    results = retrieve(
        query,
        strategy=strategy,
        top_k=top_k,
    )


    # ========================================================
    # No Results
    # ========================================================

    if not results:

        return {
            "answer": (
                "I don't know based on the available "
                "knowledge base."
            ),
            "fallback": True,
            "similarity": 0.0,
            "sources": [],
        }


    # ========================================================
    # Top-1 Similarity
    # ========================================================

    top_similarity = (
        results[0]["similarity"]
    )


    # ========================================================
    # Similarity-Based Fallback
    # ========================================================

    if top_similarity < threshold:

        return {
            "answer": (
                "I don't know based on the available "
                "knowledge base."
            ),
            "fallback": True,
            "similarity": top_similarity,
            "sources": [],
        }


    # ========================================================
    # Build Retrieved Context
    # ========================================================

    context_parts = []


    for result in results:

        context_parts.append(
            f"Source: {result['source']}\n"
            f"Title: {result['title']}\n"
            f"Content: {result['text']}"
        )


    context = "\n\n".join(
        context_parts
    )


    # ========================================================
    # MOCK_LLM PATH
    # ========================================================

    if MOCK_LLM:

        answer = mock_llm_answer(
            query=query,
            results=results,
        )


        return {
            "answer": answer,
            "fallback": False,
            "similarity": top_similarity,
            "sources": [
                result["source"]
                for result in results
            ],
        }


    # ========================================================
    # REAL LLM PATH
    # ========================================================

    llm = get_llm()


    system_prompt = """
You are a banking and financial customer-support assistant.

Answer the user's question ONLY using the supplied
knowledge-base context.

Rules:

1. Do not use outside knowledge.

2. Do not invent facts, policies, numbers,
   fees, requirements, or procedures.

3. Do not make assumptions that are not supported
   by the supplied context.

4. If the context does not contain enough information
   to answer the question, respond exactly with:

I don't know based on the available knowledge base.

5. Give a concise and clear customer-support answer.

6. Do not mention that you are an AI unless necessary.
"""


    user_prompt = f"""
Knowledge-base context:

{context}

User question:

{query}

Answer using ONLY the knowledge-base context above.
"""


    response = llm.invoke(
        [
            SystemMessage(
                content=system_prompt
            ),
            HumanMessage(
                content=user_prompt
            ),
        ]
    )


    answer = (
        response.content.strip()
    )


    return {
        "answer": answer,
        "fallback": False,
        "similarity": top_similarity,
        "sources": [
            result["source"]
            for result in results
        ],
    }


# ============================================================
# Main Test
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # Calibrate Threshold
    # ========================================================

    threshold = (
        calibrate_threshold()
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "RETRIEVAL TEST"
    )

    print(
        "=" * 70
    )


    print(
        f"\nMOCK_LLM: {MOCK_LLM}"
    )

    print(
        f"Calibrated threshold: "
        f"{threshold:.4f}"
    )


    # ========================================================
    # Test Queries
    # ========================================================

    test_queries = [

        "What documents are required for KYC?",

        "How is EMI calculated?",

        "What factors affect credit score?",

        "What are the rules for prepayment?",

        "What are NRI account eligibility requirements?",

        "What is the weather forecast today?",
    ]


    # ========================================================
    # Execute Tests
    # ========================================================

    for query in test_queries:

        print(
            "\n" + "-" * 70
        )

        print(
            f"Query: {query}"
        )


        result = grounded_answer(
            query,
            strategy="sentence",
            top_k=3,
            threshold=threshold,
        )


        print(
            f"Top-1 similarity: "
            f"{result['similarity']:.4f}"
        )


        print(
            f"Threshold: "
            f"{threshold:.4f}"
        )


        print(
            f"Fallback: "
            f"{result['fallback']}"
        )


        print(
            f"Sources: "
            f"{result['sources']}"
        )


        print(
            f"Answer:\n"
            f"{result['answer']}"
        )