FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
# Install PyTorch CPU first to reduce image size and ensure compatibility
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# Manually install sentence-transformers to resolve import error
RUN pip install --no-cache-dir sentence-transformers==2.2.2
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port (Render sets PORT env var, we'll use a script or CMD to handle it)
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
