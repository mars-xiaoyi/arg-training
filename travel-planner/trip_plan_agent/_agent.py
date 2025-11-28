import json
import logging
from typing import List
from llms import get_chat_model
from schemas import InitTravelPlanRequest, Venue
from .trip_plan_prompts import TRIP_PLAN_PROMPT
from .rag_utils import RAGUtility
from pydantic import ValidationError
# import config # REMOVED THIS LINE

class TripPlanAgent:
    def __init__(self):
        self.chat_model = get_chat_model()
        self.conversation_history = [
            {"role": "system", "content": TRIP_PLAN_PROMPT},
        ]
        self.rag_utility = RAGUtility()

    def run(self, request: InitTravelPlanRequest) -> List[Venue]:
        logging.info(f"TripPlanAgent received request and start querying venues.")
        user_request_str = request.model_dump_json()

        # Retrieve information using RAGUtility, passing natural_language_query from the request
        retrieved_venues: List[Venue] = self.rag_utility.retrieve_information(request, request.natural_language_query)

        logging.info(f"TripPlanAgent found {len(retrieved_venues)} venues, start generating trip plan.")

        # Format retrieved venues for the LLM prompt using model_dump_json()
        if retrieved_venues:
            retrieved_context = "\n\nRelevant Venues from Knowledge Base (JSON format):\n"
            # Convert each Venue object to a JSON string and join them
            retrieved_context += "[\n" + ",\n".join([venue.model_dump_json(indent=2) for venue in retrieved_venues]) + "\n]"
        else:
            retrieved_context = "\nNo specific venues found based on your request and query."

        # Construct the prompt with user preferences and retrieved context
        prompt_content = f"User Preferences: {user_request_str}"
        if request.natural_language_query: # Use natural_language_query from request
            prompt_content += f"\nUser Query: {request.natural_language_query}"
        prompt_content += retrieved_context

        self.conversation_history.append({"role": "user", "content": prompt_content})

        response_message = self.chat_model.generate_response(
            self.conversation_history,
            response_format={"type": "json_object"},
        )

        response_content = response_message.content
        venue_dicts = []
        try:
            # Strip markdown code block fences if present
            if response_content.startswith("```json") and response_content.endswith("```"):
                response_content = response_content[len("```json"):-len("```")].strip()
            # Attempt to decode the LLM's response as JSON
            venue_dicts = json.loads(response_content)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON response from LLM: {e}")
            logging.error(f"LLM response content: {response_content}")
            return [] # Return an empty list if JSON decoding fails

        venues = []
        if isinstance(venue_dicts, list):
            # Attempt to convert each dictionary to a Venue object, handling validation errors
            for venue_dict in venue_dicts:
                try:
                    venues.append(Venue(**venue_dict))
                except ValidationError as e:
                    logging.warning(f"Validation error for a venue dictionary from LLM: {e}")
                    logging.warning(f"Invalid venue dictionary: {venue_dict}")
                except TypeError as e:
                    logging.warning(f"Type error when creating Venue object from LLM response: {e}")
                    logging.warning(f"Problematic item: {venue_dict}")
        else:
            logging.error(f"LLM response was not a list of venues as expected. Received type: {type(venue_dicts)}")
            logging.error(f"LLM response content: {response_content}")
            return []

        logging.info(f"Generated {len(venues)} venues from LLM response.")
        return venues