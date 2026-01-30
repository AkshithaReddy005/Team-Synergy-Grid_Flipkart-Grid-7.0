FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 8001

# Default command: generate sessions, train G-BERT, then start API
CMD ["python", "train_and_run.py", "--catalog", "../SRP-main/Dataset_Final_TeamSynergyGrid.csv"]
