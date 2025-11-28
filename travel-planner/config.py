import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CHAT_MODEL_PROVIDER = "google"  # Can be 'google' or 'azure'

class RAGConfig:
    """Configuration for RAG components."""
    EMBEDDING_MODEL_NAME: str = "text-embedding-004" # Configurable embedding model, e.g., "text-embedding-004"
    KNOWLEDGE_BASE_PATH: str = os.path.join(os.path.dirname(__file__), "travel_knowledge_base")
    TOP_K_RETRIEVAL: int = 5 # Number of top results to retrieve from vector search