from schemas import Venue

def generate_venue_schema_description() -> str:
    """Generates a human-readable description of the Venue Pydantic schema."""
    schema = Venue.model_json_schema()
    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    description_lines = []
    for field_name, field_props in properties.items():
        # Removed unused field_type assignment and integrated type into description
        field_type = field_props.get("type", "any")
        field_description = field_props.get("description", "No description provided.")
        
        if field_name in required_fields:
            # Add field type to required field description
            description_lines.append(f"- \"{field_name}\": ({field_type}) {field_description}")
        else:
            # Add field type to optional field description
            description_lines.append(f"- \"{field_name}\": (Optional, {field_type}) {field_description}")
    return "\n".join(description_lines)

TRIP_PLAN_PROMPT = f"""
You are a Destination Knowledge Agent. Your core responsibility is to assist in creating a travel plan based on user preferences and a list of relevant venues.
You will receive structured requests from a Preference Agent, containing the user's travel preferences in JSON format.
Additionally, you will be provided with a list of relevant venues, also in JSON format, retrieved from a knowledge base.

Your task is to use the user's preferences and the provided relevant venues to generate a coherent and appealing travel plan.
The travel plan should be presented as a JSON array of Venue objects, where each venue is selected and potentially enhanced with details relevant to a travel itinerary.
You should only provide information and not get involved in complex decision-making or optimization beyond creating a travel plan from the given information.

Your output MUST be a JSON array of Venue objects, conforming to the Venue Pydantic schema.
Each JSON object in the array should represent a venue and have the following structure:
{generate_venue_schema_description()}
"""