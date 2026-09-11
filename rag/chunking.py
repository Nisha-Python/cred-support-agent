from pathlib import Path
import re


KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"


def load_documents():
    """
    Load all Markdown knowledge-base documents.

    Returns:
        list[dict]: Each document contains:
            - doc_id
            - title
            - text
            - source
    """
    documents = []

    for file_path in sorted(KB_DIR.glob("*.md")):
        text = file_path.read_text(encoding="utf-8").strip()

        title = file_path.stem.replace("_", " ").title()

        documents.append(
            {
                "doc_id": file_path.stem,
                "title": title,
                "text": text,
                "source": file_path.name,
            }
        )

    if len(documents) < 12:
        raise ValueError(
            f"Expected at least 12 KB documents, found {len(documents)}"
        )

    return documents


def fixed_size_chunks(
    text: str,
    chunk_size: int = 350,
    overlap: int = 70,
):
    """
    Split text into fixed-size character chunks with overlap.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size"
        )

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def sentence_chunks(
    text: str,
    sentences_per_chunk: int = 2,
):
    """
    Split text into chunks containing a fixed number of sentences.
    """
    if sentences_per_chunk <= 0:
        raise ValueError("sentences_per_chunk must be positive")

    # Split on sentence-ending punctuation.
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    chunks = []

    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(
            sentences[i:i + sentences_per_chunk]
        )

        if chunk:
            chunks.append(chunk)

    return chunks


def create_chunks():
    """
    Create both required chunking strategies.

    Returns:
        tuple:
            fixed_chunks
            sentence_based_chunks
    """
    documents = load_documents()

    fixed_chunks = []
    sentence_based_chunks = []

    for document in documents:

        # Strategy 1: fixed-size with overlap
        chunks_1 = fixed_size_chunks(
            document["text"],
            chunk_size=350,
            overlap=70,
        )

        for index, chunk in enumerate(chunks_1):
            fixed_chunks.append(
                {
                    "chunk_id": (
                        f"{document['doc_id']}_fixed_{index}"
                    ),
                    "doc_id": document["doc_id"],
                    "title": document["title"],
                    "source": document["source"],
                    "chunk_index": index,
                    "text": chunk,
                    "strategy": "fixed_size_overlap",
                }
            )

        # Strategy 2: sentence-based
        chunks_2 = sentence_chunks(
            document["text"],
            sentences_per_chunk=2,
        )

        for index, chunk in enumerate(chunks_2):
            sentence_based_chunks.append(
                {
                    "chunk_id": (
                        f"{document['doc_id']}_sentence_{index}"
                    ),
                    "doc_id": document["doc_id"],
                    "title": document["title"],
                    "source": document["source"],
                    "chunk_index": index,
                    "text": chunk,
                    "strategy": "sentence_based",
                }
            )

    return fixed_chunks, sentence_based_chunks


if __name__ == "__main__":
    fixed_chunks, sentence_chunks_result = create_chunks()

    print("=" * 60)
    print("CHUNKING TEST")
    print("=" * 60)

    print(f"Knowledge-base documents: {len(load_documents())}")
    print(f"Fixed-size chunks: {len(fixed_chunks)}")
    print(
        f"Sentence-based chunks: "
        f"{len(sentence_chunks_result)}"
    )

    print("\nExample fixed-size chunk:")
    print("-" * 60)
    print(fixed_chunks[0]["text"])

    print("\nExample sentence-based chunk:")
    print("-" * 60)
    print(sentence_chunks_result[0]["text"])

    print("\nChunking test: PASSED")
    print("=" * 60)