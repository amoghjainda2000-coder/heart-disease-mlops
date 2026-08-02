# 1. Use lightweight official Python runtime base image
FROM python:3.11-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy requirements first (leverages Docker caching layer)
COPY requirements.txt .

# 4. Install dependencies without caching build artifacts
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy API code and trained model artifacts into container
COPY src/ ./src/
COPY models/ ./models/

# 6. Expose the port FastAPI runs on
EXPOSE 8000

# 7. Command to run the application using Uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]