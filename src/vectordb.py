import os
import chromadb
import uuid
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        """
        Simple text chunking by splitting on spaces and grouping into chunks.

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk
            chunk_overlap: Number of overlapping characters between chunks

        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        # Split text into words
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            # If adding this word would exceed chunk size and we have some content
            if current_length + word_length > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(" ".join(current_chunk))
                
                # Start new chunk with overlap
                if chunk_overlap > 0:
                    # Calculate how many words to keep for overlap (rough estimate)
                    overlap_words = max(1, int(chunk_overlap / (current_length / len(current_chunk))))
                    current_chunk = current_chunk[-overlap_words:]
                    current_length = sum(len(word) + 1 for word in current_chunk) - 1
                else:
                    current_chunk = []
                    current_length = 0
            
            # Add word to current chunk
            current_chunk.append(word)
            current_length += word_length
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def add_documents(self, documents: List[Dict]) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents with 'content' and 'metadata' keys
        """
        print(f"Processing {len(documents)} documents...")
        
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for doc_idx, document in enumerate(documents):
            content = document.get("content", "")
            metadata = document.get("metadata", {})
            
            if not content:
                continue
                
            # Chunk the document content
            chunks = self.chunk_text(content)
            
            for chunk_idx, chunk in enumerate(chunks):
                # Create unique ID
                chunk_id = f"doc_{doc_idx}_chunk_{chunk_idx}"
                
                # Create chunk metadata
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_id": chunk_idx,
                    "total_chunks": len(chunks),
                    "document_id": doc_idx
                })
                
                all_chunks.append(chunk)
                all_metadatas.append(chunk_metadata)
                all_ids.append(chunk_id)
        
        if not all_chunks:
            print("No valid chunks found to add to database.")
            return
        
        # Generate embeddings for all chunks
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = self.embedding_model.encode(all_chunks).tolist()
        
        # Add to ChromaDB collection
        print("Adding chunks to vector database...")
        self.collection.add(
            embeddings=embeddings,
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )
        
        print(f"Successfully added {len(all_chunks)} chunks to vector database")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results
        """
        if not query.strip():
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "ids": [],
            }
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Search in ChromaDB
        try:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            return {
                "documents": results.get("documents", [[]])[0],
                "metadatas": results.get("metadatas", [[]])[0],
                "distances": results.get("distances", [[]])[0],
                "ids": results.get("ids", [[]])[0],
            }
        except Exception as e:
            print(f"Error during search: {e}")
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "ids": [],
            }