from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from app.services.analyzer_service import analyze_match

router = APIRouter()

class MatchRequest(BaseModel):
    resume_text: str
    job_description: str

class MatchResponse(BaseModel):
    score: int
    missing_keywords: List[str]
    match_level: str

@router.post("/match", response_model=MatchResponse)
async def match_resume(payload: MatchRequest):
    """
    Analyze the match between a resume text and a job description.
    """
    if not payload.resume_text or not payload.job_description:
        raise HTTPException(status_code=400, detail="Both resume text and job description are required")
    
    result = analyze_match(payload.resume_text, payload.job_description)
    return result
