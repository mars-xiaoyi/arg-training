import os
import faiss
import numpy as np
import re
import uuid
import json
from typing import Optional, Dict
from langchain_google_vertexai import VertexAIEmbeddings

from schemas import Venue
from config import RAGConfig

class KnowledgeBaseManager:
    """
    Manages the loading of the knowledge base from files and building the FAISS index.
    """
    def __init__(self):
        self.embeddings = VertexAIEmbeddings(model_name=RAGConfig.EMBEDDING_MODEL_NAME)
        self.knowledge_base: Dict[str, Venue] = {}
        self.destination_faiss_index = None
        self.venue_id_map = {}
        self._load_knowledge_base_from_files()
        if self.knowledge_base:
            self._build_faiss_index()
    
    def _load_knowledge_base_from_files(self):
        """
        Loads knowledge base from Markdown files with YAML front matter.
        Each file is expected to represent a destination and contain multiple venues.
        """
        print(f"Loading knowledge base from: {RAGConfig.KNOWLEDGE_BASE_PATH}")
        if not os.path.exists(RAGConfig.KNOWLEDGE_BASE_PATH):
            print(f"Knowledge base path not found: {RAGConfig.KNOWLEDGE_BASE_PATH}")
            return

        for filename in os.listdir(RAGConfig.KNOWLEDGE_BASE_PATH):
            # Minimal change: Switch from .md to .json file check
            if filename.endswith(".json"):
                file_path = os.path.join(RAGConfig.KNOWLEDGE_BASE_PATH, filename)
                destination_name = os.path.splitext(filename)[0]
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # Minimal change: Load JSON instead of reading markdown text
                        venue_data_list = json.load(f)
                        # Minimal change: Iterate over JSON venue dicts instead of splitting markdown
                        for venue_data in venue_data_list:
                            venue = self._parse_venue_details(venue_data, destination_name)
                            if venue:
                                self.knowledge_base[venue.id] = venue
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        print(f"Loaded {len(self.knowledge_base)} venues into knowledge base.")

    def _build_faiss_index(self):
        """
        Builds a FAISS index from the descriptions of the loaded venues.
        """
        if not self.knowledge_base:
            print("No knowledge base content to build index.")
            return

        texts_to_embed = []
        venue_ids = [] # List to store venue IDs in the order of embeddings
        for venue_id, venue in self.knowledge_base.items(): # Iterate through dictionary items
            texts_to_embed.append(venue.description)
            venue_ids.append(venue_id) # Store the venue ID

        if not texts_to_embed:
            print("No descriptions found to embed for FAISS index.")
            return

        print(f"Generating embeddings for {len(texts_to_embed)} venue descriptions...")
        embeddings_list = self.embeddings.embed_documents(texts_to_embed)
        embeddings_array = np.array(embeddings_list).astype("float32")

        dimension = embeddings_array.shape[1]
        self.destination_faiss_index = faiss.IndexFlatL2(dimension)
        self.destination_faiss_index.add(embeddings_array)
        print(f"FAISS index built with {self.destination_faiss_index.ntotal} entries.")

        # Create a mapping from FAISS index to venue ID
        self.venue_id_map = {i: venue_ids[i] for i in range(len(venue_ids))}
        print("Venue ID map created.")

    def _parse_venue_details(self, venue_data: dict, destination: str) -> Optional[Venue]:
        """
        Parses a dictionary (representing a venue) and extracts details into a Venue object.
        """
        try:
            # Generate a unique ID for the venue if not already present
            venue_id = venue_data.get("id", str(uuid.uuid4()))

            return Venue(
                id=venue_id,
                name=venue_data.get("name", "Unknown Name"),
                type=venue_data.get("type", "Unknown Type"),
                description=venue_data.get("description", "No description available."),
                geo_location=venue_data.get("geo_location"),
                address=venue_data.get("address"),
                opening_hours=venue_data.get("opening_hours"),
                rating=venue_data.get("rating"),
                budget=venue_data.get("budget"),
                suitable_for=venue_data.get("suitable_for"),
                tags=venue_data.get("tags"),
                destination=destination
            )
        except Exception as e:
            # Use venue_data.get("name") for more robust error message
            print(f"Error creating Venue object for '{venue_data.get('name', 'Unknown Venue')}' in '{destination}': {e}")
            return None