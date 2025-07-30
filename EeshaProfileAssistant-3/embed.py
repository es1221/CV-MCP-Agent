import os
import json
import logging
import numpy as np
from typing import List, Tuple, Dict, Any
import faiss
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from markitdown import MarkItDown
import pickle
import hashlib
from datetime import datetime

class DocumentEmbedder:
    def __init__(self):
        """Initialize document embedder with FAISS index"""
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.dimension = 1536  # text-embedding-3-small dimension
        self.index = None
        self.documents = []
        self.metadata = []
        
        # Paths for persistence
        self.index_path = "data/faiss_index.bin"
        self.docs_path = "data/documents.pkl"
        self.metadata_path = "data/metadata.pkl"
        self.hash_path = "data/doc_hashes.json"
        
        # Try to load existing index or create new one
        if not self._load_index():
            self._load_documents()
            self._create_index()
            self._save_index()
    
    def _get_file_hash(self, filepath: str) -> str:
        """Get hash of file content for change detection"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""
    
    def _check_documents_changed(self) -> bool:
        """Check if any documents have changed since last indexing"""
        try:
            if not os.path.exists(self.hash_path):
                return True
                
            with open(self.hash_path, 'r') as f:
                old_hashes = json.load(f)
            
            # Check each document
            doc_files = ['data/cv.pdf', 'data/masters_dissertation.pdf']
            for doc_file in doc_files:
                if os.path.exists(doc_file):
                    current_hash = self._get_file_hash(doc_file)
                    if old_hashes.get(doc_file) != current_hash:
                        return True
            
            return False
        except:
            return True
    
    def _save_document_hashes(self):
        """Save hashes of indexed documents"""
        hashes = {}
        doc_files = ['data/cv.pdf', 'data/masters_dissertation.pdf']
        for doc_file in doc_files:
            if os.path.exists(doc_file):
                hashes[doc_file] = self._get_file_hash(doc_file)
        
        with open(self.hash_path, 'w') as f:
            json.dump(hashes, f)
    
    def _load_index(self) -> bool:
        """Load existing FAISS index and documents from disk"""
        try:
            # Check if all necessary files exist
            if not all(os.path.exists(p) for p in [self.index_path, self.docs_path, self.metadata_path]):
                logging.info("Index files not found, will create new index")
                return False
            
            # Check if documents have changed
            if self._check_documents_changed():
                logging.info("Documents have changed, will recreate index")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(self.index_path)
            
            # Load documents and metadata
            with open(self.docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            logging.info(f"Loaded existing FAISS index with {self.index.ntotal} documents")
            return True
            
        except Exception as e:
            logging.error(f"Error loading index: {str(e)}")
            return False
    
    def _save_index(self):
        """Save FAISS index and documents to disk"""
        try:
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
            
            # Save documents and metadata
            with open(self.docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            # Save document hashes
            self._save_document_hashes()
            
            logging.info(f"Saved FAISS index with {self.index.ntotal} documents")
            
        except Exception as e:
            logging.error(f"Error saving index: {str(e)}")
    
    def _load_documents(self):
        """Load documents from data directory"""
        try:
            # Load CV - check for cached markdown first
            cv_text = self._load_file_cached('data/cv.pdf', 'data/cv.md')
            if cv_text:
                chunks = self._chunk_text(cv_text, 'CV')
                self.documents.extend([chunk[0] for chunk in chunks])
                self.metadata.extend([chunk[1] for chunk in chunks])
            
            # Load dissertation - check for cached markdown first
            dissertation_text = self._load_file_cached('data/masters_dissertation.pdf', 'data/masters_dissertation.md')
            if dissertation_text:
                chunks = self._chunk_text(dissertation_text, 'Masters Dissertation')
                self.documents.extend([chunk[0] for chunk in chunks])
                self.metadata.extend([chunk[1] for chunk in chunks])
            

            
            logging.info(f"Loaded {len(self.documents)} document chunks")
            
        except Exception as e:
            logging.error(f"Error loading documents: {str(e)}")
            # Create dummy documents for testing
            self.documents = [
                "Eesha Sondhi is a skilled software engineer with experience in Python, machine learning, and web development.",
                "She has worked on various projects including AI applications and data analysis tools.",
                "Her educational background includes a Master's degree in Computer Science with focus on AI."
            ]
            self.metadata = [
                {'source': 'CV', 'chunk_id': 0},
                {'source': 'CV', 'chunk_id': 1},
                {'source': 'CV', 'chunk_id': 2}
            ]
    
    def _load_file_cached(self, pdf_path: str, md_path: str) -> str:
        """Load text content from file, using cached markdown if available"""
        try:
            # Check if cached markdown exists and is newer than PDF
            if os.path.exists(md_path) and os.path.exists(pdf_path):
                pdf_mtime = os.path.getmtime(pdf_path)
                md_mtime = os.path.getmtime(md_path)
                
                if md_mtime > pdf_mtime:
                    # Use cached markdown
                    logging.info(f"Using cached markdown: {md_path}")
                    with open(md_path, 'r', encoding='utf-8') as f:
                        return f.read()
            
            # Process PDF with markitdown and cache result
            if os.path.exists(pdf_path):
                logging.info(f"Processing PDF with markitdown: {pdf_path}")
                md = MarkItDown()
                result = md.convert(pdf_path)
                
                # Save processed content to markdown file
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(result.text_content)
                logging.info(f"Cached processed content to: {md_path}")
                
                return result.text_content
            
            return ""
            
        except Exception as e:
            logging.error(f"Error loading file {pdf_path}: {str(e)}")
            return ""

    def _load_file(self, filepath: str) -> str:
        """Load text content from file using markitdown for better PDF processing"""
        try:
            if filepath.endswith('.txt') or filepath.endswith('.md'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return f.read()
            elif filepath.endswith('.pdf'):
                # Use markitdown for enhanced PDF processing
                md = MarkItDown()
                result = md.convert(filepath)
                return result.text_content
            else:
                logging.warning(f"Unsupported file type: {filepath}")
                return ""
        except FileNotFoundError:
            logging.warning(f"File not found: {filepath}")
            return ""
        except Exception as e:
            logging.error(f"Error loading file {filepath}: {str(e)}")
            return ""
    
    def _chunk_text(self, text: str, source: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Semantic-aware chunking using LangChain RecursiveCharacterTextSplitter"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # character count (not tokens)
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""],  # priority for where to split
        )
        
        docs = splitter.create_documents([text])
        chunks = []
        
        for i, doc in enumerate(docs):
            metadata = {
                'source': source,
                'chunk_id': i
            }
            chunks.append((doc.page_content, metadata))
        
        return chunks
    
    def _create_index(self):
        """Create FAISS index from documents"""
        try:
            if not self.documents:
                logging.warning("No documents to index")
                return
            
            # Generate embeddings for all documents
            embeddings = []
            batch_size = 100  # Process in batches to avoid rate limits
            
            for i in range(0, len(self.documents), batch_size):
                batch = self.documents[i:i + batch_size]
                logging.info(f"Generating embeddings for batch {i//batch_size + 1}/{(len(self.documents) + batch_size - 1)//batch_size}")
                
                for doc in batch:
                    embedding = self._get_embedding(doc)
                    embeddings.append(embedding)
            
            # Create FAISS index
            embeddings_array = np.array(embeddings).astype('float32')
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_array)
            self.index.add(embeddings_array)
            
            logging.info(f"Created FAISS index with {self.index.ntotal} documents")
            
        except Exception as e:
            logging.error(f"Error creating index: {str(e)}")
            # Create empty index as fallback
            self.index = faiss.IndexFlatIP(self.dimension)
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using OpenAI API"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logging.error(f"Error getting embedding: {str(e)}")
            # Return zero embedding as fallback
            return np.zeros(self.dimension)
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (text, score, source) tuples
        """
        try:
            if not self.index or self.index.ntotal == 0:
                logging.warning("Index is empty")
                return []
            
            # Get query embedding
            query_embedding = self._get_embedding(query)
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.documents):
                    text = self.documents[idx]
                    source = self.metadata[idx]['source']
                    results.append((text, float(score), source))
            
            return results
            
        except Exception as e:
            logging.error(f"Error in search: {str(e)}")
            return []
