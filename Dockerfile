FROM python:3.8-slim

WORKDIR /app

# Ensure pip is up to date and install dependencies globally
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# Copy application artifacts and source code
COPY model.pkl .
COPY scaler.pkl .
COPY app.py .

EXPOSE 8000

# Run uvicorn via python module
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]