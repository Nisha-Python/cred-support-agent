from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from chunking import create_chunks


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

CHROMA_DIR = PROJECT_DIR / "chroma_db"

MODEL_NAME = "all-MiniLM-L6-v2"


# ------------------------------------------------------------
# Load embedding model
# ------------------------------------------------------------

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

print(f"Embedding model: {MODEL_NAME}")


# ------------------------------------------------------------
# Create ChromaDB client
# ------------------------------------------------------------

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)


# ------------------------------------------------------------
# Index a chunk collection
# ------------------------------------------------------------

def index_chunks(collection_name, chunks):
    """
    Create a ChromaDB collection and index all chunks.
    """

    # Delete existing collection if it exists so that
    # re-running the script produces deterministic results.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine"
        }
    )

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "doc_id": chunk["doc_id"],
            "title": chunk["title"],
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "strategy": chunk["strategy"],
        }
        for chunk in chunks
    ]

    print(
        f"Creating embeddings for {len(documents)} chunks..."
    )

    embeddings = model.encode(
        documents,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )

    return collection


# ------------------------------------------------------------
# Main indexing process
# ------------------------------------------------------------

def build_indexes():

    fixed_chunks, sentence_chunks = create_chunks()

    print("\n" + "=" * 60)
    print("BUILDING CHROMADB INDEXES")
    print("=" * 60)

    print("\n1. Fixed-size + overlap collection")

    fixed_collection = index_chunks(
        "loan_kb_fixed",
        fixed_chunks,
    )

    print("\n2. Sentence-based collection")

    sentence_collection = index_chunks(
        "loan_kb_sentence",
        sentence_chunks,
    )

    print("\n" + "=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)

    print(
        f"Fixed collection chunks: "
        f"{fixed_collection.count()}"
    )

    print(
        f"Sentence collection chunks: "
        f"{sentence_collection.count()}"
    )

    print(f"ChromaDB directory: {CHROMA_DIR}")

    print("\nEmbedding and indexing: PASSED")


if __name__ == "__main__":
    build_indexes()