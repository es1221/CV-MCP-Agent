#!/usr/bin/env python3
"""
Pre-generate embeddings for all documents

Run this script before starting the application to ensure embeddings
are generated and cached, avoiding regeneration on each server start.
"""

import os
import logging
from embed import DocumentEmbedder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """Generate and save embeddings for all documents"""
    print("Pre-generating embeddings for CV Agent...")
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Check if required files exist
    required_files = ['data/cv.pdf', 'data/masters_dissertation.pdf']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Warning: Missing files: {missing_files}")
        print("Continuing with available files...")
    
    # Initialize embedder (this will generate and save embeddings)
    print("Initializing document embedder...")
    embedder = DocumentEmbedder()
    
    # Test the embeddings with a sample query
    print("\nTesting embeddings with sample query...")
    results = embedder.search("Eesha Sondhi experience", top_k=3)
    
    if results:
        print(f"\nFound {len(results)} results:")
        for i, (text, score, source) in enumerate(results, 1):
            print(f"\n{i}. Source: {source} (Score: {score:.3f})")
            print(f"   {text[:100]}...")
    else:
        print("No results found - check if documents were loaded correctly")
    
    print("\nEmbeddings pre-generation complete!")
    print("The following files have been created:")
    print("  - data/faiss_index.bin (FAISS index)")
    print("  - data/documents.pkl (Document texts)")
    print("  - data/metadata.pkl (Document metadata)")
    print("  - data/doc_hashes.json (Document hashes for change detection)")

if __name__ == "__main__":
    main()