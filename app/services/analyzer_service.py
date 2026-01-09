import re
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, before_log

# Configure Logging
logger = logging.getLogger(__name__)

def extract_keywords(text: str) -> set:
    # Simple keyword extraction (in a real app, use spaCy or NLTK)
    # Allows alphanumeric and simple hyphens (e.g. "pre-filled")
    words = re.findall(r'\b[a-zA-Z0-9-]+\b', text.lower())
    # FIlter out common stop words (very basic list)
    stop_words = {"and", "the", "to", "in", "of", "a", "is", "for", "with", "on", "at", "by", "an", "be"}
    return {w for w in words if w not in stop_words and len(w) > 2}

@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before=before_log(logger, logging.INFO)
)
def analyze_match(resume_text: str, job_description: str) -> dict:
    """
    Analyzes match with exponential backoff retry for robustness.
    Simulates AI service call which might fail intermittently.
    """
    resume_keywords = extract_keywords(resume_text)
    job_keywords = extract_keywords(job_description)
    
    if not job_keywords:
        return {"score": 0, "missing_keywords": [], "match_level": "Unknown"}

    # Find intersection
    matched = resume_keywords.intersection(job_keywords)
    
    # Calculate score
    score = int((len(matched) / len(job_keywords)) * 100)
    
    # Identify missing important keywords (simple diff)
    missing = list(job_keywords - resume_keywords)[:10]  # Top 10 missing
    
    match_level = "Low"
    if score > 70:
        match_level = "High"
    elif score > 40:
        match_level = "Medium"
        
    return {
        "score": score,
        "missing_keywords": missing,
        "match_level": match_level
    }
