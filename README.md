# Nexus AI Backend (Enterprise Edition)

## 🚀 Overview
AI-Powered Career Intelligence API built with **FastAPI**.
Features industry-standard "Enterprise" patterns:
- **Rate Limiting:** Protects against abuse (SlowAPI).
- **Robust Retries:** Handles AI service flakes (Tenacity).
- **Tracing:** Request ID generation for distributed tracing (ASGI Correlation ID).
- **Dockerized:** Ready for container deployment.

## 🛠️ Tech Stack
- **Framework:** FastAPI
- **Language:** Python 3.10
- **Validation:** Pydantic V2
- **Testing:** Pytest

## ⚡ Deployment
### GitHub Actions
The `.github/workflows/ci.yml` pipeline automatically tests the code on every push.

### Render.com
1. Create a **Web Service**.
2. Connect this repository.
3. Runtime: **Docker**.
4. Deploy!

## 🧪 Local Development
```bash
# Run with Docker
docker-compose up

# Run manually
pip install -r requirements.txt
uvicorn app.main:app --reload
```
