# Company Report Assistant, deployment image
# Uses requirements-cloud.txt deliberately, not the full local
# requirements.txt, since the deployed container never runs Ollama
# or a local embedding model. Keeping torch, sentence-transformers,
# and Chroma out of this image entirely keeps builds fast and the
# image small, they are only needed for local development.

FROM python:3.13-slim

WORKDIR /app

# Install dependencies first, separately from copying the rest of
# the code, so Docker can cache this layer and skip reinstalling
# packages on every code change during future rebuilds.
COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt

# Only copy what the deployed app actually needs. The source PDFs
# and local vector_store are deliberately left out, they are only
# used by local development (Chroma path), not by the deployed
# Pinecone-based cloud path.
COPY app.py .
COPY pipeline/ ./pipeline/
COPY data/chunks_export/ ./data/chunks_export/

# Render sets the PORT environment variable at runtime, Streamlit
# needs to bind to it and to 0.0.0.0 to be reachable from outside
# the container.
CMD ["sh", "-c", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"]