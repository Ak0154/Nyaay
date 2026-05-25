from fastapi import APIRouter
from pydantic import BaseModel

from services.retrieval import retrieve_context
from services.LLM import generate_answer

router = APIRouter()


class QueryRequest(BaseModel):

    query: str
    document_id: str


@router.post("/ask")
async def ask(data: QueryRequest):

    context, sources = retrieve_context(
        data.query,
        data.document_id
    )

    answer = generate_answer(
        data.query,
        context
    )

    return {

        "success": True,
        "answer": answer,
        "sources": sources

    }