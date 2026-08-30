"""
test_hf_embedding_api.py

One-off diagnostic script. Tests whether Hugging Face's hosted
Inference API actually supports BAAI/bge-base-en-v1.5 for embeddings
on a free account, before wiring it into the real retrieval code.

Run from the project root:
    python diagnostics/test_hf_embedding_api.py
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
MODEL_NAME = "BAAI/bge-base-en-v1.5"

if not HF_API_TOKEN:
    raise ValueError("HF_API_TOKEN not found. Check your .env file.")

print(f"Testing Hugging Face Inference API with model: {MODEL_NAME}")
print("Sending a test sentence to embed...\n")

client = InferenceClient(token=HF_API_TOKEN)

try:
    result = client.feature_extraction(
        "What percentage of Sun Pharma's revenue comes from its US business?",
        model=MODEL_NAME,
    )
    print("SUCCESS.")
    print(f"Returned embedding with {len(result)} dimensions.")
    print(f"First 5 values: {result[:5] if hasattr(result, '__getitem__') else 'N/A'}")
except Exception as e:
    print("FAILED.")
    print(f"Error: {e}")
