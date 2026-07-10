"""
Run this once to (re)build the vector index from documents in data/sample_docs.
Usage: python ingest.py
"""

from rag_pipeline import build_index

if __name__ == "__main__":
    build_index()
