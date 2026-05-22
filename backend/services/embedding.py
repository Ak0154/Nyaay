import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------

BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_FILE = BASE_DIR /"chunks"/ "chunks.json"
CHROMA_PATH = BASE_DIR / "chroma_db"

# ---------------- LOAD MODEL ----------------

print("Loading embedding model...")

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

# ---------------- LOAD CHUNKS ----------------

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")

# ---------------- CHROMA ----------------

client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = client.get_or_create_collection(
    name="legal_chunks"
)

# ---------------- EMBEDDINGS ----------------

texts = [chunk["text"] for chunk in chunks]

print("Generating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True
)

# ---------------- STORE ----------------

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

print("Done.")
print(
    f"Stored {collection.count()} chunks in Chroma"
)