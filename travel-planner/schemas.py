from pydantic import BaseModel, Field
from typing import List, Optional

class InitTravelPlanRequest(BaseModel):
    destination: str = Field(..., description="The desired travel destination.")
    travel_dates: Optional[str] = Field(None, description="Specific travel dates or period (e.g., 'summer 2024', 'next month').")
    number_of_travelers: Optional[int] = Field(None, description="The number of people traveling.")
    interests: Optional[List[str]] = Field(None, description="A list of interests or activities the user enjoys (e.g., 'hiking', 'museums', 'food tours').")
    budget: Optional[str] = Field(None, description="The user's budget preference (e.g., 'economy', 'mid-range', 'luxury').")
    preferred_venue_types: Optional[List[str]] = Field(None, description="Specific types of venues the user is interested in (e.g., 'restaurant', 'hotel', 'attraction').")
    natural_language_query: Optional[str] = Field(None, description="A natural language query for semantic search, providing additional context or specific preferences beyond structured fields.")

class Venue(BaseModel):
    id: str = Field(..., description="A unique identifier for the venue.") # New ID field
    name: str = Field(..., description="The name of the venue.")
    type: str = Field(..., description="The type of venue (e.g., 'attraction', 'restaurant', 'hotel', 'event', 'service').")
    description: str = Field(..., description="A brief description of the venue.")
    geo_location: Optional[str] = Field(None, description="The geographical location or coordinates of the venue.")
    address: Optional[str] = Field(None, description="The physical address of the venue.")
    opening_hours: Optional[str] = Field(None, description="The operating hours of the venue (e.g., '9 AM - 5 PM', '24/7').")
    rating: Optional[float] = Field(None, description="The average rating of the venue.")
    budget: Optional[str] = Field(None, description="The budget level of the venue (e.g., 'economy', 'mid-range', 'luxury').")
    suitable_for: Optional[List[str]] = Field(None, description="A list of suitable demographics or interests (e.g., ['families', 'couples']).")
    tags: Optional[List[str]] = Field(None, description="A list of relevant tags (e.g., ['historical', 'outdoor']).")
    destination: str = Field(..., description="The city or destination where the venue is located")