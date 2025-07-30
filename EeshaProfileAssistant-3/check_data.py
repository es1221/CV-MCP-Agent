#!/usr/bin/env python3
"""
Check if all data files are present and loaded correctly
"""

import os
import json
import pickle

def check_files():
    """Check all required data files"""
    print("Checking data files...\n")
    
    # Check embeddings files
    embedding_files = [
        "data/faiss_index.bin",
        "data/documents.pkl",
        "data/metadata.pkl",
        "data/doc_hashes.json"
    ]
    
    print("Embedding files:")
    for file in embedding_files:
        exists = os.path.exists(file)
        print(f"  {file}: {'✓ EXISTS' if exists else '✗ MISSING'}")
        if exists and file.endswith('.pkl'):
            try:
                with open(file, 'rb') as f:
                    data = pickle.load(f)
                    print(f"    → Contains {len(data)} items")
            except Exception as e:
                print(f"    → Error loading: {str(e)}")
    
    print("\nJSON files:")
    # Check JSON files
    json_files = [
        "data/repo_summaries.json",
        "data/deseng.json",
        "data/personal.json"
    ]
    
    for file in json_files:
        exists = os.path.exists(file)
        print(f"  {file}: {'✓ EXISTS' if exists else '✗ MISSING'}")
        if exists:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        print(f"    → Contains {len(data)} entries")
                        # Show first few keys
                        keys = list(data.keys())[:3]
                        for key in keys:
                            print(f"      - {key}")
            except Exception as e:
                print(f"    → Error loading: {str(e)}")
    
    print("\nPDF files:")
    # Check PDF files
    pdf_files = [
        "data/cv.pdf",
        "data/masters_dissertation.pdf"
    ]
    
    for file in pdf_files:
        exists = os.path.exists(file)
        print(f"  {file}: {'✓ EXISTS' if exists else '✗ MISSING'}")
        
        # Check cached markdown
        md_file = file.replace('.pdf', '.md')
        md_exists = os.path.exists(md_file)
        print(f"  {md_file}: {'✓ EXISTS' if md_exists else '✗ MISSING'}")

if __name__ == "__main__":
    check_files()