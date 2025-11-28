import config # Import config at the very top to ensure logging is configured
from trip_plan_agent import TripPlanAgent
from schemas import InitTravelPlanRequest
import pytest

# Reusable fixture to initialize agent ONCE (avoids redundant knowledge base loading)
@pytest.fixture(scope="module")
def trip_plan_agent():
    agent = TripPlanAgent()
    yield agent

@pytest.mark.asyncio
async def test_trip_plan_agent_case1_paris_family_mid_range(trip_plan_agent):
    request = InitTravelPlanRequest(
        destination="Paris",
        number_of_travelers=4,
        budget="mid-range",
        natural_language_query="Family-friendly cultural sightseeing in Paris",
        interests=["sightseeing", "culture", "family-friendly", "art", "history", "outdoor"]
    )
    response = trip_plan_agent.run(request)
    print(f"\n--- Response for Test Case 1 (Paris, Family, Mid-range) ---")
    if response:
        print(f"Number of venues returned: {len(response)}")
        for i, venue in enumerate(response):
            print(f"  {i+1}. Name: {venue.name}, ID: {venue.id}")
    else:
        print("No venues returned.")

@pytest.mark.asyncio
async def test_trip_plan_agent_case2_london_history_economy(trip_plan_agent):
    request = InitTravelPlanRequest(
        destination="London",
        number_of_travelers=1,
        budget="economy",
        natural_language_query="Budget historical sites and free activities in London",
        interests=["history", "museums", "landmarks", "free", "outdoor", "culture"]
    )
    response = trip_plan_agent.run(request)
    print(f"\n--- Response for Test Case 2 (London, History, Economy) ---")
    if response:
        print(f"Number of venues returned: {len(response)}")
        for i, venue in enumerate(response):
            print(f"  {i+1}. Name: {venue.name}, ID: {venue.id}")
    else:
        print("No venues returned.")

@pytest.mark.asyncio
async def test_trip_plan_agent_case3_tokyo_entertainment_mid_range(trip_plan_agent):
    request = InitTravelPlanRequest(
        destination="Tokyo",
        number_of_travelers=2,
        budget="mid-range",
        natural_language_query="Cultural entertainment and food in Tokyo",
        interests=["entertainment", "culture", "food", "urban", "shopping", "art"]
    )
    response = trip_plan_agent.run(request)
    print(f"\n--- Response for Test Case 3 (Tokyo, Entertainment, Mid-range) ---")
    if response:
        print(f"Number of venues returned: {len(response)}")
        for i, venue in enumerate(response):
            print(f"  {i+1}. Name: {venue.name}, ID: {venue.id}")
    else:
        print("No venues returned.")

@pytest.mark.asyncio
async def test_trip_plan_agent_case4_paris_luxury_culture(trip_plan_agent):
    request = InitTravelPlanRequest(
        destination="Paris",
        number_of_travelers=2,
        budget="luxury",
        natural_language_query="Luxury cultural experiences in Paris",
        interests=["luxury", "culture", "history", "art", "royal", "gardens"]
    )
    response = trip_plan_agent.run(request)
    print(f"\n--- Response for Test Case 4 (Paris, Luxury Culture) ---")
    if response:
        print(f"Number of venues returned: {len(response)}")
        for i, venue in enumerate(response):
            print(f"  {i+1}. Name: {venue.name}, ID: {venue.id}")
    else:
        print("No venues returned.")

@pytest.mark.asyncio
async def test_trip_plan_agent_case6_london_parks_nature(trip_plan_agent):
    request = InitTravelPlanRequest(
        destination="London",
        number_of_travelers=1,
        budget="economy",
        natural_language_query="Quiet parks and nature spots in London",
        interests=["parks", "gardens", "quiet places", "nature", "relaxation", "free", "outdoor"]
    )
    response = trip_plan_agent.run(request)
    print(f"\n--- Response for Test Case 6 (London, Parks & Nature) ---")
    if response:
        print(f"Number of venues returned: {len(response)}")
        for i, venue in enumerate(response):
            print(f"  {i+1}. Name: {venue.name}, ID: {venue.id}")
    else:
        print("No venues returned.")

# Empty response test cases (kept as required)
@pytest.mark.asyncio
async def test_trip_plan_agent_case7_non_existent_destination(trip_plan_agent):
    request = InitTravelPlanRequest(
        destination="Berlin",
        number_of_travelers=2,
        budget="economy",
        natural_language_query="Sightseeing in Berlin",
        interests=["sightseeing"]
    )
    response = trip_plan_agent.run(request)
    print(f"\n--- Response for Test Case 7 (Non-existent Destination) ---")
    if response:
        print(f"Number of venues returned: {len(response)}")
        for i, venue in enumerate(response):
            print(f"  {i+1}. Name: {venue.name}, ID: {venue.id}")
    else:
        print("No venues returned.")

@pytest.mark.asyncio
async def test_trip_plan_agent_case8_unlikely_combination_tokyo(trip_plan_agent):
    request = InitTravelPlanRequest(
        destination="Tokyo",
        number_of_travelers=1,
        budget="luxury",
        natural_language_query="Ancient ruins and cowboy culture in Tokyo",
        interests=["ancient ruins", "cowboy culture", "western"]
    )
    response = trip_plan_agent.run(request)
    print(f"\n--- Response for Test Case 8 (Unlikely Combination, Tokyo) ---")
    if response:
        print(f"Number of venues returned: {len(response)}")
        for i, venue in enumerate(response):
            print(f"  {i+1}. Name: {venue.name}, ID: {venue.id}")
    else:
        print("No venues returned.")

@pytest.mark.asyncio
async def test_trip_plan_agent_case9_paris_no_match_budget(trip_plan_agent):
    request = InitTravelPlanRequest(
        destination="Paris",
        number_of_travelers=2,
        budget="super-economy", # Budget not in KB
        natural_language_query="Ultra cheap places in Paris",
        interests=["food"]
    )
    response = trip_plan_agent.run(request)
    print(f"\n--- Response for Test Case 9 (Paris, No Match Budget) ---")
    if response:
        print(f"Number of venues returned: {len(response)}")
        for i, venue in enumerate(response):
            print(f"  {i+1}. Name: {venue.name}, ID: {venue.id}")
    else:
        print("No venues returned.")

@pytest.mark.asyncio
async def test_trip_plan_agent_case10_non_existent_destination(trip_plan_agent):
    request = InitTravelPlanRequest(
        destination="Mars", # Non-existent destination
        number_of_travelers=1,
        budget="economy",
        natural_language_query="Things to do on Mars",
        interests=["space"]
    )
    response = trip_plan_agent.run(request)
    print(f"\n--- Response for Test Case 10 (Non-existent Destination) ---")
    if response:
        print(f"Number of venues returned: {len(response)}")
        for i, venue in enumerate(response):
            print(f"  {i+1}. Name: {venue.name}, ID: {venue.id}")
    else:
        print("No venues returned.")