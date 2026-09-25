"""Ingest the eight policy documents into ChromaDB.

Uses local sentence-transformer embeddings with all-MiniLM-L6-v2.
The script is deterministic and does not require an LLM API key.
"""

from pathlib import Path
import os


# ---------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------
# Disable Chroma's anonymized product telemetry for this local project.
# This must be set before importing chromadb.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

# Windows may not support Hugging Face cache symlinks without
# Developer Mode/admin privileges. The cache still works without them.
os.environ.setdefault(
    "HF_HUB_DISABLE_SYMLINKS_WARNING",
    "1",
)


import chromadb
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
DB = ROOT / "chroma_db"

COLLECTION = "zepto_policy"
MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_documents():
    """Load the eight required policy documents."""

    paths = sorted(
        DOCS.glob("doc_*.txt"),
        key=lambda p: p.name
    )

    if len(paths) != 8:
        raise RuntimeError(
            f"Expected exactly 8 policy documents in {DOCS}, "
            f"but found {len(paths)}."
        )

    documents = []

    for path in paths:

        text = path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            raise RuntimeError(
                f"Document is empty: {path.name}"
            )

        documents.append(
            {
                "id": f"{path.stem}#0",
                "text": text,
                "metadata": {
                    "document_id": path.stem,
                    "chunk_id": 0,
                    "source": path.name,
                },
            }
        )

    return documents


def get_collection(client):
    """Create a fresh ChromaDB collection for reproducible ingestion."""

    try:
        client.delete_collection(COLLECTION)
        print(
            f"Removed existing ChromaDB collection: {COLLECTION}"
        )
    except Exception:
        # Collection may not exist on the first run.
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={
            "hnsw:space": "cosine"
        },
    )

    return collection


# ---------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------
def build_collection():
    """Load, embed, and index all eight policy documents."""

    print("Starting Support Assistant ingestion...")
    print(f"Documents directory: {DOCS}")
    print(f"ChromaDB directory:   {DB}")
    print(f"Embedding model:      {MODEL_NAME}")

    # Validate source documents.
    documents = load_documents()

    print(
        f"Validated {len(documents)} policy documents."
    )

    # Persistent local ChromaDB client.
    client = chromadb.PersistentClient(
        path=str(DB)
    )

    collection = get_collection(client)

    # Local embedding model.
    print(
        "\nLoading local embedding model..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    ids = [
        item["id"]
        for item in documents
    ]

    texts = [
        item["text"]
        for item in documents
    ]

    metadatas = [
        item["metadata"]
        for item in documents
    ]

    # One document per chunk is acceptable for the supplied short corpus.
    # Normalize embeddings because the collection uses cosine distance.
    print(
        f"Generating embeddings for {len(texts)} documents..."
    )

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = embeddings.tolist()

    # Sanity checks before insertion.
    if not (
        len(ids)
        == len(texts)
        == len(metadatas)
        == len(embeddings)
    ):
        raise RuntimeError(
            "Embedding/index lengths do not match."
        )

    # Store vectors + source text + metadata.
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    indexed_count = collection.count()

    if indexed_count != 8:
        raise RuntimeError(
            f"Expected 8 indexed chunks, "
            f"but ChromaDB reports {indexed_count}."
        )

    # Verify that retrieval works.
    verification = collection.query(
        query_embeddings=[embeddings[0]],
        n_results=1,
    )

    if not verification.get("documents"):
        raise RuntimeError(
            "ChromaDB retrieval verification failed."
        )

    print("\nIngestion completed successfully.")
    print(
        f"Indexed documents: {indexed_count}"
    )
    print(
        f"Collection:        {COLLECTION}"
    )
    print(
        f"ChromaDB path:     {DB}"
    )

    return indexed_count


# ---------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    build_collection()