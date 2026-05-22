import chromadb
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

# Load existing Chroma DB
client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="legal_chunks"
)

while True:
    query = input("\nAsk: ")

    if query.lower() == "exit":
        break

    # Convert question into embedding
    query_embedding = model.encode(query)

    # Search
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3
    )

    print("\nTop Results:\n")

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    for i, (doc, meta) in enumerate(
        zip(docs, metas)
    ):

        print(f"Result {i+1}")
        print(f"Source: {meta['source']}")
        print(f"Page: {meta['page']}")
        print(doc[:300])
        print("-"*50)