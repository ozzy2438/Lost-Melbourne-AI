# Lost Melbourne — Retrieval Demo
# Builds a container with the Streamlit app and all retrieval dependencies.
# Requires pre-built indexes (artifacts/retrieval/) and processed corpus
# (data/processed/) to be present at build time or mounted at runtime.
#
# Build:
#   docker build -t lost-melbourne .
#
# Run (with pre-built artifacts on the host):
#   docker run -p 8501:8501 \
#     -v "$(pwd)/artifacts:/app/artifacts:ro" \
#     -v "$(pwd)/data:/app/data:ro" \
#     lost-melbourne
#
# Then open http://localhost:8501

FROM python:3.12-slim

WORKDIR /app

# Install OS-level dependencies required by torch/transformers wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for layer caching
COPY requirements-retrieval.txt requirements-app.txt ./

# Install all Python dependencies
RUN pip install --no-cache-dir \
        -r requirements-retrieval.txt \
        -r requirements-app.txt

# Copy application source
COPY src/ src/
COPY app/ app/
COPY evaluation/ evaluation/
COPY reports/retrieval_results.json reports/retrieval_results.json

# Expose Streamlit default port
EXPOSE 8501

# Streamlit config: disable the "Deploy" button and CORS warnings in containers
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0"]
