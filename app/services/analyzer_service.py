from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, before_log

# Configure Logging
logger = logging.getLogger(__name__)

def extract_keywords(text: str) -> set:
    words = re.findall(r'\b[a-zA-Z0-9-]+\b', text.lower())
    stop_words = {"and", "the", "to", "in", "of", "a", "is", "for", "with", "on", "at", "by", "an", "be"}
    return {w for w in words if w not in stop_words and len(w) > 2}

@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before=before_log(logger, logging.INFO)
)
def analyze_match(resume_text: str, job_description: str) -> dict:
    """
    Analyzes match using Hybrid Approach:
    1. Keyword Overlap (Hard Skills)
    2. Semantic Cosine Similarity (Context)
    """
    # 1. Keyword Matching
    resume_keywords = extract_keywords(resume_text)
    job_keywords = extract_keywords(job_description)
    
    keyword_score = 0
    missing = []
    
    if job_keywords:
        matched = resume_keywords.intersection(job_keywords)
        keyword_score = int((len(matched) / len(job_keywords)) * 100)
        missing = list(job_keywords - resume_keywords)[:10]

    # 2. Semantic Matching (TF-IDF Cosine Similarity)
    # Checks how similar the "meaning" or "vocabulary distribution" is
    vectors = TfidfVectorizer().fit_transform([resume_text, job_description])
    cosine_sim = cosine_similarity(vectors)[0][1]
    semantic_score = int(cosine_sim * 100)
    
    # Weighted Final Score (60% Keywords, 40% Semantic)
    final_score = int((keyword_score * 0.6) + (semantic_score * 0.4))
    
    match_level = "Low"
    if final_score > 75:
        match_level = "High"
    elif final_score > 50:
        match_level = "Medium"
        
    return {
        "score": final_score,
        "semantic_score": semantic_score,
        "keyword_score": keyword_score,
        "missing_keywords": missing,
        "match_level": match_level
    }
