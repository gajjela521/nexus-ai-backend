from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
from app.services.rag_service import rag_service

router = APIRouter()

class IngestRequest(BaseModel):
    documents: List[Dict[str, str]]  # [{"content": "...", "metadata": {...}}]

class QueryRequest(BaseModel):
    query: str
    openai_api_key: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    context_used: List[str]

@router.post("/ingest")
async def ingest_knowledge(payload: IngestRequest):
    """
    Admin Endpoint: Upload text (job descriptions, policy docs) into the Vector DB.
    """
    if not payload.documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    
    count = rag_service.ingest_documents(payload.documents)
    return {"status": "success", "chunks_indexed": count}

@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(payload: QueryRequest):
    """
    User Endpoint: Ask a question.
    Flow: User -> Embed -> Vector Search -> LLM -> Answer.
    Optionally provide 'openai_api_key' to use GPT-4 for this session.
    """
    result = rag_service.query(payload.query, api_key=payload.openai_api_key)
    return {
        "answer": result["answer"],
        "context_used": result["context_used"]
    }
