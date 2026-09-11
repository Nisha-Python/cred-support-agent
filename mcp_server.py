from fastmcp import FastMCP

from tools.loan_status import check_loan_application_status
from rag.retrieval import retrieve


mcp = FastMCP("Cred Support MCP")


@mcp.tool
def get_loan_status(record_id: str) -> dict:
    """Get the status and escalation information for a loan application."""
    return check_loan_application_status(record_id)


@mcp.tool
def search_knowledge_base(query: str) -> dict:
    """Search the Cred support knowledge base for relevant information."""

    retrieved = retrieve(
        query,
        strategy="sentence",
        top_k=3,
    )

    # The existing retriever returns a list of results.
    results = retrieved if isinstance(retrieved, list) else retrieved.get(
        "results", []
    )

    top_similarity = None

    if results:
        similarities = []

        for item in results:
            if isinstance(item, dict):
                similarity = item.get("similarity")

                if similarity is not None:
                    similarities.append(float(similarity))

        if similarities:
            top_similarity = max(similarities)

    return {
        "query": query,
        "results": results,
        "top_similarity": top_similarity,
        "fallback": (
            top_similarity is not None
            and top_similarity < 0.4569
        ),
    }
if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8001,
        path="/mcp",
    )