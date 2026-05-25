from LLM import generate_answer
import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

# Connect to Chroma
client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="legal_chunks"
)

query = input("Ask: ")

query_embedding = model.encode(query)

results = collection.query(
    query_embeddings=[
        query_embedding.tolist()
    ],
    n_results=10
)

# for i, doc in enumerate(results["documents"][0]):
#     print(f"\nResult {i+1}")
#     print(doc[:300])

context = "\n\n".join(
    results["documents"][0]
)

# print("\n========== RETRIEVED CONTEXT ==========\n")
# print(context)
# print("\n=======================================\n")

answer = generate_answer(
    query,
    context
)

print("\nAnswer:\n")
print(answer)