from app.api.v1.endpoints import scanner, rag

api_router = APIRouter()
api_router.include_router(scanner.router, prefix="/scanner", tags=["scanner"])
api_router.include_router(rag.router, prefix="/rag", tags=["rag"])
