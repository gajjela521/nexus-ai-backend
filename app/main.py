from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from asgi_correlation_id import CorrelationIdMiddleware

from app.core.config import settings
from app.api.v1.router import api_router

# Initialize Rate Limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    description="AI-Powered Career Intelligence Backend (Enterprise Edition)"
)

# Enterprise Middleware: Request ID Tracing
app.add_middleware(CorrelationIdMiddleware)

# Register Limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/health")
@limiter.limit("10/minute")
def health_check(request: Request):
    return {"status": "ok", "app_name": settings.PROJECT_NAME}

@app.get("/")
def root():
    return {"message": "Welcome to Nexus AI Backend. Visit /docs for documentation."}
