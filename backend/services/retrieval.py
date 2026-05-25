from services.embedding import (
    model,
    collection
)

def retrieve_context(
    query: str,
    document_id: str
):

    query_embedding = model.encode(
        query
    )

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],

        n_results=5,

        where={
            "document_id":
            document_id
        }
    )

    docs = results["documents"][0]

    metadata = results[
        "metadatas"
    ][0]

    context = "\n\n".join(
        docs
    )

    return context, metadata