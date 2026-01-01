FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory
WORKDIR $HOME/serve

# Copy requirements
COPY --chown=user requirements.txt .

# Install Python dependencies as user 
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user . .

# Expose HF Spaces port
EXPOSE 7860

# Run FastAPI
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "7860"]
