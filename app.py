import os
import json
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
#for document processing
import PyPDF2
from bs4 import BeautifulSoup
import markdown
#for vector operations
import numpy as np
import faiss
print("all libraries successfully imported")

@dataclass
class Chunk:
    """Text chunk with metadata and embedding."""
    id: str
    text: str
    vector: Optional[np.ndarray]
    metadata: Dict

class DocumentLoader:
    """load PDF, Markdown, and HTML documents."""
    @staticmethod
    def load_pdf(file_path: str) -> List[Dict]:
        """extract text from PDF, page by page for citations."""
        chunks = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        chunks.append({
                            'text': text,
                            'metadata': {
                                'source': os.path.basename(file_path),
                                'page': page_num + 1,
                                'type': 'pdf'
                            }
                        })
        except Exception as e:
            print(f"error loading PDF {file_path}: {e}")
        return chunks
    
    @staticmethod
    def load_markdown(file_path: str) -> List[Dict]:
        """convert markdown to text via HTML"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                md_content = file.read()
                html = markdown.markdown(md_content)
                soup = BeautifulSoup(html, 'html.parser')
                text = soup.get_text()

                return [{
                    'text': text,
                    'metadata': {
                        'source': os.path.basename(file_path),
                        'page': 1,
                        'type': 'markdown'
                    }
                }]
        except Exception as e:
            print(f"error loading markdown {file_path}: {e}")
            return []
    
    @staticmethod
    def load_html(file_path: str) -> List[Dict]:
        """extract text from HTML, removing scripts and styles."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
                for script in soup(["script", "style"]):
                    script.decompose()
                    text = soup.get_text()

                    return [{
                        'text': text,
                        'metadata': {
                            'source': os.path.basename(file_path),
                            'page': 1,
                            'type': 'html'
                        }
                    }]
        except Exception as e:
            print(f"error loading HTML {file_path}: {e}")
            return []
        
    @staticmethod
    def load_documents(directory: str) -> List[Dict]:
        """load all supported documents from a directory."""
        documents = []
        doc_dir = Path(directory)

        if not doc_dir.exists():
            print(f"Creating {directory}...")
            doc_dir.mkdir(parents=True)
            print(f"add documents to directory {directory} and run again.")
            return documents
        
        for file_path in doc_dir.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()

                if ext == '.pdf':
                    documents.extend(DocumentLoader.load_pdf(str(file_path)))
                elif ext in ['.md', '.markdown']:
                    documents.extend(DocumentLoader.load_markdown(str(file_path)))
                elif ext in ['.html', '.htm']:
                    documents.extend(DocumentLoader.load_html(str(file_path)))
            print(f"loaded {len(documents)} document sections")
            return documents
    print("document loader ready!")

class TextChunker:
    """text chunking with overlap and sentence boundaries."""
    @staticmethod
    def clean_text(text: str) -> str:
        """normalize whitespace and remove special characters."""
        text = re.sub(r'\s+', '', text) #multiple spaces to single space
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;]', '', text) #keep punctuation
        return text.strip()
        
    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 750,
        overlap: int = 100,
        metadata: Dict = None
    ) -> List[Chunk]:
        """split text into overlapping chunks at sentence boundaries.
        Args:
            chunk_size: target size (≈150 tokens for embeddings)
            overlap: overlap size to preserve context
            metadata: source info for citations
        """
        text = TextChunker.clean_text(text)
        chunks = []

        if not text:
            return chunks
            
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size

            #break at sentence boundary(last 20% of chunk)
            if end < len(text):
                search_start = end - int(chunk_size * 0.2)
                sentence_end = max(
                    text.rfind('.', search_start, end),
                    text.rfind('!', search_start, end),
                    text.rfind('?', search_start,end)
                )
            
                if sentence_end != -1 and sentence_end > start:
                    end = sentence_end + 1

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata['chunk_index'] = chunk_index
                chunk_id = f"{chunk_metadata.get('source', 'unknown')}_{chunk_index}"

                chunks.append(Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    vector=None,
                    metadata=chunk_metadata
                ))

                chunk_index += 1
            start = end - overlap #move with overlap
            if start >= len(text) - overlap:
                break
        return chunks
print("text chunker ready!")

class OllamaEmbedder:
    """Generate embeddings using Ollama's embedding model."""
    def __init__(self, model_name: str = "nomic-embed-text"):
        self.embedding_cache = {}
        self.model_name = model_name
        self._verify_model()

    def _verify_model(self):
        """check if model is available locally."""
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                check=True
            )
            if self.model_name not in result.stdout:
                raise RuntimeError(
                    f"Model '{self.model_name} not found locally.\n'"
                    f"Please download it first using:\n"
                    f"  ollama pull {self.model_name}\n"
                    f"This is a one-time setup that requries internet connection."
                )
            print(f"Found embedding model: {self.model_name}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama service.\n"
                f"Please ensure Ollama is installed and running.\n"
                f"Error: {e}"
            )
        except FileNotFoundError:
            raise RuntimeError(
                "Ollama not found on your system.\n"
                "Please install Ollama from: https://ollama.com/download\n"
                "This is a one-time setup step."
            )
        
    def embed_text(self, text: str) -> np.ndarray:
        # Check cache first
        if text in self.embedding_cache:
            print("✓ Using cached embedding")  # Nice feedback!
            return self.embedding_cache[text]
    
        try:
            import http.client
            conn = http.client.HTTPConnection("localhost", 11434, timeout=30)
            headers = {'Content-Type': 'application/json'}
            payload = json.dumps({
                "model": self.model_name,
                "prompt": text
            })

            conn.request("POST", "/api/embeddings", payload, headers)
            response = conn.getresponse()
            data = json.loads(response.read().decode())
        
            # Cache the SAME NumPy array you return
            embedding = np.array(data['embedding'], dtype=np.float32)
            self.embedding_cache[text] = embedding  # FIXED: Cache NumPy array
            return embedding
        
        except Exception as e:
            print(f"Embedding error: {e}")
            return np.zeros(768, dtype=np.float32)  # Fallback

        
    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """generate embeddings for all chunks with progress."""
        print(f"Generating embeddings for {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks):
            if i % 10 == 0 and i > 0:
                print(f" progress: {i}/{len(chunks)}")
            chunk.vector = self.embed_text(chunk.text)
        
        print("embeddings complete!")
        return chunks
    
class VectorDatabase:
    """FAISS-based vector storage and retrieval with Cosine Similarity."""
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        #use IndexFlatIP for cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[Chunk] = []

    def add_chunks(self, chunks: List[Chunk]):
        """add chunk embeddings to index."""
        vectors = np.array([chunk.vector for chunk in chunks], dtype=np.float32)

        #normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)

        self.index.add(vectors)
        self.chunks.extend(chunks)
        print(f"added {len(chunks)} chunks (total: {len(self.chunks)})")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """find top-k most similar chunks using cosine similarity.
        Returns:
            List of (chunk, distance) tuples
            Distance is (1 - cosine_similarity), so lower = more similar
        """
        query_vector = query_vector.reshape(1, -1).astype(np.float32)

        #normalize query vector for cosine similarity
        faiss.normalize_L2(query_vector)

        #search (returns similarity scores, not distances)
        similarities, indices = self.index.search(query_vector, top_k)

        results = []
        for idx,similarity in zip(indices[0], similarities[0]):
            if idx < len(self.chunks):
                #convert similarity to distance: distance = 1 - similarity
                distance = 1 - similarity
                results.append((self.chunks[idx], float(distance)))

        return results

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
    
        # Save FAISS index (unchanged)
        faiss.write_index(self.index, os.path.join(directory, 'faiss.index'))
    
        # Save chunks metadata (unchanged)  
        chunks_data = [{'id': chunk.id, 'text': chunk.text, 'metadata': chunk.metadata} for chunk in self.chunks]
        with open(os.path.join(directory, 'chunks.json'), 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2)

        # NEW: Save embeddings as binary NumPy file (FAST)
        vectors = np.stack([chunk.vector for chunk in self.chunks])  # Stack into 2D array
        np.save(os.path.join(directory, 'vectors.npy'), vectors)     # Save binary
        print(f"database saved to {directory}")

    def load(self, directory: str, embedder=None):  # embedder now optional!
        index_path = os.path.join(directory, 'faiss.index')
        chunks_path = os.path.join(directory, 'chunks.json')
        vectors_path = os.path.join(directory, 'vectors.npy')  # NEW

        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            print(f"no database found in {directory}")
            return False
    
        # Load FAISS index (unchanged)
        self.index = faiss.read_index(index_path)
    
        # Load chunks metadata (unchanged)
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
    
        self.chunks = []
    
        # NEW: Check if we have pre-saved embeddings
        if os.path.exists(vectors_path):
            print("Loading pre-computed embeddings (FAST)...")
            vectors = np.load(vectors_path)  # Load binary embeddings
            for i, data in enumerate(chunks_data):
                chunk = Chunk(
                    id=data['id'],
                    text=data['text'], 
                    vector=vectors[i],      # Use saved embedding!
                    metadata=data['metadata']
                )
                self.chunks.append(chunk)
        else:
            # Fallback: re-embed (your old slow code)
            print("Recomputing embeddings (SLOW fallback)...")
            for data in chunks_data:
                chunk = Chunk(
                    id=data['id'],
                    text=data['text'],
                    vector=embedder.embed_text(data['text']),  # Only if no .npy
                    metadata=data['metadata']
                )
                self.chunks.append(chunk)
    
        print(f"database loaded: {len(self.chunks)} chunks")
        return True


class OllamaLLM:
    """LLM interface using Ollama CLI (more reliable on CPU)."""

    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        self.response_cache = {}  # NEW: Cache full answers
        self._verify_model()

    def _verify_model(self):
        """check if model is available locally."""
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                check=True
            )
            if self.model_name not in result.stdout:
                raise RuntimeError(
                    f"Model '{self.model_name}' not found locally.\n"
                    f"Please download it first using:\n"
                    f" ollama pull {self.model_name}\n"
                    f"This is a one-time setup step that requires internet connection."
                )
            print(f"Found LLM model: {self.model_name}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama service.\n"
                f"Please ensure Ollama is installed and running.\n"
                f"Error: {e}"
            )
        except FileNotFoundError:
            raise RuntimeError(
                "Ollama not found on your system.\n"
                "Please install Ollama from: https://ollama.com/download\n"
                "This is a one-time setup step."
            )

    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        """generate response using Ollama CLI (more reliable on CPU)
        Args:
            prompt: Complete prompt with context and question
            temperature: Creativity (0.0=deterministic, 1.0=creative)
        """
        # CACHE HIT = INSTANT
        if prompt in self.response_cache:
            print("🚀 Using cached LLM response")
            return self.response_cache[prompt]
        try:
            print(f" Generating with {self.model_name} ...")

            import http.client
            import json

            conn = http.client.HTTPConnection("localhost", 11434, timeout=120)
            payload = json.dumps({
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature}
            })

            conn.request("POST", "/api/generate", payload, {'Content-Type': 'application/json'})
            response = conn.getresponse()
            data = json.loads(response.read().decode())

            answer = data['response'].strip()
            print(f"Generated {len(answer)} characters")
            
            # CACHE MISS = Save for next time
            self.response_cache[prompt] = answer
            print(f" Generated {len(answer)} characters")
            
            return answer
        
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f" {error_msg}")
            return error_msg

class RAGSystem:
    """complete RAG orchestration."""
    def __init__(
        self,
        documents_dir: str = "documents",
        db_dir: str = "vector_db",
        llm_model: str = "llama3.2",
        embedding_model: str = "nomic-embed-text"
    ):
        self.documents_dir = documents_dir
        self.db_dir = db_dir
        
        print("initializing RAG System...")
        self.embedder = OllamaEmbedder(embedding_model)
        self.llm = OllamaLLM(llm_model)
        self.vector_db = VectorDatabase()
        print("RAG System initialized!")
    
    def ingest_documents(
        self,
        chunk_size: int = 750,
        overlap: int = 100,
        force_rebuild: bool = False
    ):
        """Build or load vector database."""    
        #try loading existing database
        if not force_rebuild and os.path.exists(self.db_dir):
            print("loading existing database...")
            if self.vector_db.load(self.db_dir, self.embedder):
                return
        
        print(" Building new database...")
        
        #load documents
        documents = DocumentLoader.load_documents(self.documents_dir)
        if not documents:
            print("no documents found!")
            return
        
        #chunk documents
        all_chunks = []
        for doc in documents:
            chunks = TextChunker.chunk_text(
                doc['text'],
                chunk_size=chunk_size,
                overlap=overlap,
                metadata=doc['metadata']
            )
            all_chunks.extend(chunks)
        
        print(f"created {len(all_chunks)} chunks")       
        #generate embeddings
        all_chunks = self.embedder.embed_chunks(all_chunks)       
        #store in vector DB
        self.vector_db.add_chunks(all_chunks)
        #save for future use
        self.vector_db.save(self.db_dir)
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        distance_threshold: float = 1.5
    ) -> Dict:
        """Answer question using RAG.
       Returns:
            {
                'answer': Generated answer,
                'sources': List of source chunks,
                'confidence': 'high'|'medium'|'low'
            }
        """
        print(f"\n Question: {question}")
        
        #embed query
        query_vector = self.embedder.embed_text(question)
        
        #search vector DB
        results = self.vector_db.search(query_vector, top_k=top_k)
        
        #filter by threshold
        filtered_results = [
            (chunk, dist) for chunk, dist in results
            if dist < distance_threshold
        ]
        
        if not filtered_results:
            return {
                'answer': "Insufficient context to answer this question.",
                'sources': [],
                'confidence': 'low'
            }
        
        #build context from chunks
        context_parts = []
        sources = []
        
        for i, (chunk, distance) in enumerate(filtered_results):
            context_parts.append(
                f"[Source {i+1}: {chunk.metadata['source']}, "
                f"Page {chunk.metadata.get('page', 'N/A')}]\n{chunk.text}\n"
            )
            sources.append({
                'id': chunk.id,
                'source': chunk.metadata['source'],
                'page': chunk.metadata.get('page', 'N/A'),
                'distance': distance
            })
        
        context = "\n".join(context_parts)
        
        #build prompt
        prompt = f"""You are a helpful AI assistant. Answer the question based ONLY on the provided context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based only on the context above
2. Cite source numbers (e.g., "According to Source 1...")
3. If context is insufficient, state that clearly
4. Be concise but thorough

ANSWER:"""       
        #generate answer
        print("Generating answer...")
        answer = self.llm.generate(prompt, temperature=0.3)
        
        return {
            'answer': answer,
            'sources': sources,
            'confidence': 'high' if len(filtered_results) >= 3 else 'medium'
        }

print("RAG System class ready!")


# Initialize RAG system
rag = RAGSystem(
    documents_dir="documents",
    db_dir="vector_db",
    llm_model="llama3.2",
    embedding_model="nomic-embed-text"
)

# Save at bottom of script
rag.ingest_documents(force_rebuild=False)
while True:
    question = input("Ask: ")
    result = rag.query(question)
    print(result['answer'])


# Display results
print("\n" + "="*60)
print("ANSWER:")
print("="*60)
print(result['answer'])
print("\n" + "="*60)
print(f"CONFIDENCE: {result['confidence'].upper()}")
print("="*60)
print("\nSOURCES:")
for i, source in enumerate(result['sources'], 1):
    print(f"  {i}. {source['source']} (Page {source['page']}) - Distance: {source['distance']:.4f}")
print("="*60)