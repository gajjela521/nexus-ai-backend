FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
# Install PyTorch CPU first to reduce image size and ensure compatibility
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install sentence-transformers

COPY . .

# Expose port (Render sets PORT env var, we'll use a script or CMD to handle it)
EXPOSE 10000

# Command to run the application
CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"
