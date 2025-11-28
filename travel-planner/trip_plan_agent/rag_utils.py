import numpy as np
from typing import List, Optional, Set
from schemas import InitTravelPlanRequest, Venue
from .knowledge_base_manager import KnowledgeBaseManager
from config import RAGConfig

class RAGUtility:
    def __init__(self):
        # Initialize KnowledgeBaseManager to load the knowledge base and FAISS index
        kb_manager = KnowledgeBaseManager()
        self.knowledge_base = kb_manager.knowledge_base # knowledge_base is now a dictionary
        self.destination_faiss_index = kb_manager.destination_faiss_index
        self.embeddings = kb_manager.embeddings # Initialize embeddings here
        self.venue_id_map = kb_manager.venue_id_map # Get the venue_id_map

    def _structured_filter_venues(self, request: InitTravelPlanRequest) -> List[Venue]:
        filtered_venues = []
        for venue in self.knowledge_base.values():
            # Filter by destination
            if request.destination and not (venue.destination and request.destination.lower() in venue.destination.lower()):
                continue

            # Filter by budget
            if request.budget and not (venue.budget and request.budget.lower() in venue.budget.lower()):
                continue

            # Filter by interests (tags and suitable_for)
            if request.interests:
                match_by_tags = venue.tags and any(interest.lower() in [tag.lower() for tag in venue.tags] for interest in request.interests)
                match_by_suitable_for = venue.suitable_for and any(interest.lower() in [s.lower() for s in venue.suitable_for] for interest in request.interests)
                if not (match_by_tags or match_by_suitable_for):
                    continue

            filtered_venues.append(venue)

        return filtered_venues

    def _semantic_search_venues(self, query: str, venues: List[Venue], top_k: int = RAGConfig.TOP_K_RETRIEVAL) -> List[Venue]:
        if not query or not venues:
            return venues

        # Convert the list of filtered venues to a set of their IDs for efficient lookup
        filtered_venue_ids: Set[str] = {venue.id for venue in venues}

        # Generate embeddings for the query
        query_embedding = self.embeddings.embed_query(query)
        query_embedding_np = np.array(query_embedding).astype('float32').reshape(1, -1)

        # Search the main FAISS index
        # We search for a larger number of results initially to ensure we get enough matches
        # after filtering by the `filtered_venue_ids`.
        search_limit = min(self.destination_faiss_index.ntotal, len(self.knowledge_base))
        D, I = self.destination_faiss_index.search(query_embedding_np, search_limit)

        # Map back to original venues using the venue_id_map and filter by the `filtered_venue_ids`
        semantic_search_results = []
        for idx in I[0]:
            if idx < len(self.knowledge_base): # Ensure index is within bounds
                venue_id = self.venue_id_map.get(idx)
                if venue_id and venue_id in filtered_venue_ids:
                    # Retrieve the actual Venue object directly from the knowledge_base dictionary
                    semantic_search_results.append(self.knowledge_base[venue_id])
            if len(semantic_search_results) >= top_k:
                break

        return semantic_search_results

    def retrieve_information(self, request: InitTravelPlanRequest, natural_language_query: Optional[str] = None) -> List[Venue]:
        # Step 1: Structured Filtering
        filtered_venues = self._structured_filter_venues(request)

        # Step 2: Semantic Search (if a natural language query is provided)
        if natural_language_query:
            semantically_filtered_venues = self._semantic_search_venues(natural_language_query, filtered_venues)
            return semantically_filtered_venues
        else:
            return filtered_venues