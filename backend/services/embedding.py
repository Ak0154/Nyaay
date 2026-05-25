import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_FILE = BASE_DIR / "chunks" / "chunks.json"
CHROMA_PATH = BASE_DIR / "chroma_db"

# Load once
model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = client.get_or_create_collection(
    name="legal_chunks"
)


def populate_chroma():

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    ids = [
        str(chunk["chunk_id"])
        for chunk in chunks
    ]

    metadatas = [
        {
            "source": chunk["source"],
            "page": str(chunk["page"]),
            "chunk_type": chunk["chunk_type"]
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )
# New function
def store_in_chroma(
    chunks,
    document_id
):

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts
    )

    ids = [
        f"{document_id}_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [

        {
    "document_id": document_id,

    "source":
    Path(
        chunk.metadata.get(
            "source",
            "unknown"
        )
    ).name,

    "page":
    str(
        chunk.metadata.get(
            "page",
            0
        )
    ),

    "chunk_type":"text"
        }

        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

if __name__ == "__main__":
    populate_chroma()