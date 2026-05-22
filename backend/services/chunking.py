"""
Chunking Legal Documents using LangChain + pdfplumber
-------------------------------------------------------
Supports: PDF (with table preservation), TXT files
Outputs:  Chunks saved as JSON with source metadata
"""

import json
import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader


# ── CONFIG ────────────────────────────────────────────────────────────────────

# DOCS_FOLDER   = "./output"
# OUTPUT_FILE   = "./chunks.json"
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_FOLDER = BASE_DIR / "output"
OUTPUT_FILE = BASE_DIR/"chunks" / "chunks.json"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50


# ── LOAD ALL DOCUMENTS ────────────────────────────────────────────────────────

def load_all_documents(folder: str) -> list[Document]:
    docs = []

    # TXT files via LangChain
    for txt_file in Path(folder).rglob("*.txt"):
        loader = TextLoader(str(txt_file), encoding="utf-8")
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["type"] = "txt"
        docs += loaded
        print(f"  📄 {txt_file.name} — loaded as text")

    print(f"\n✅ Total: {len(docs)} page/document(s) loaded\n")
    return docs


# ── CHUNK ─────────────────────────────────────────────────────────────────────

def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Split documents into chunks.
    [TABLE]...[/TABLE] blocks are kept intact — not split mid-table.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks = []

    for doc in docs:
        # Separate table blocks from regular text so tables aren't split
        parts = doc.page_content.split("[TABLE]")
        sub_docs = []

        for part in parts:
            if "[/TABLE]" in part:
                table_content, rest = part.split("[/TABLE]", 1)
                # Keep the whole table as one chunk (don't split it)
                sub_docs.append(Document(
                    page_content=f"[TABLE]\n{table_content.strip()}\n[/TABLE]",
                    metadata={**doc.metadata, "chunk_type": "table"}
                ))
                # Split remaining text normally
                if rest.strip():
                    sub_docs.append(Document(
                        page_content=rest.strip(),
                        metadata={**doc.metadata, "chunk_type": "text"}
                    ))
            else:
                if part.strip():
                    sub_docs.append(Document(
                        page_content=part.strip(),
                        metadata={**doc.metadata, "chunk_type": "text"}
                    ))

        for sub in sub_docs:
            if sub.metadata.get("chunk_type") == "table":
                all_chunks.append(sub)          # table = never split
            else:
                all_chunks += splitter.split_documents([sub])

    print(f"✅ {len(all_chunks)} chunk(s) created\n")
    return all_chunks


# ── SAVE ──────────────────────────────────────────────────────────────────────

def save_chunks(chunks: list[Document], output_file: str):
    data = [
        {
            "chunk_id":   i,
            "text":       chunk.page_content,
            "source":     chunk.metadata.get("source", "unknown"),
            "page":       chunk.metadata.get("page", "N/A"),
            "chunk_type": chunk.metadata.get("chunk_type", "text"),
        }
        for i, chunk in enumerate(chunks)
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved to '{output_file}'")


# ── PREVIEW ───────────────────────────────────────────────────────────────────

def preview_chunks(chunks: list[Document], n: int = 3):
    text_chunks  = [c for c in chunks if c.metadata.get("chunk_type") != "table"]
    table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]

    print(f"{'─'*60}")
    print(f"  📊 {len(table_chunks)} table chunk(s)  |  📝 {len(text_chunks)} text chunk(s)")
    print(f"{'─'*60}")

    for i, chunk in enumerate(chunks[:n]):
        label = "🗃️  TABLE" if chunk.metadata.get("chunk_type") == "table" else "📝 TEXT"
        print(f"\n[Chunk {i}] {label} | Source: {Path(chunk.metadata.get('source','')).name} | Page: {chunk.metadata.get('page','N/A')}")
        print(chunk.page_content[:300] + ("..." if len(chunk.page_content) > 300 else ""))

    print(f"\n{'─'*60}\n")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(DOCS_FOLDER, exist_ok=True)
    print(f"Looking in: {DOCS_FOLDER}")
    # Demo sample if folder is empty
    if not any(Path(DOCS_FOLDER).iterdir()):
        print("⚠️  No documents found. Creating sample legal text for demo...\n")
        sample = """Section 1: Right to Constitutional Remedies
The right to move the Supreme Court by appropriate proceedings for the enforcement
of the rights conferred by this Part is guaranteed.

Section 2: Protection against arbitrary arrest
No person who is arrested shall be detained in custody without being informed,
as soon as may be, of the grounds for such arrest nor shall he be denied the right
to consult, and to be defended by, a legal practitioner of his choice.
"""
        with open(f"{DOCS_FOLDER}/sample_legal.txt", "w") as f:
            f.write(sample)

    docs   = load_all_documents(DOCS_FOLDER)
    chunks = chunk_documents(docs)

    preview_chunks(chunks, n=4)
    save_chunks(chunks, OUTPUT_FILE)

    print(f"\n📦 Done! {len(chunks)} chunks ready for embedding.\n")