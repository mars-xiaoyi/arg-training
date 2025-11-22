import os
import json
from llms import get_chat_model
from prompts import PREFERENCE_CAHT_PROMPT, JSON_SUMMARY_PROMPT


class PreferenceAgent:
    """
    A travel agent that chats with the user to determine their travel preferences
    and generates a travel plan in JSON format.
    """

    def __init__(self):
        """
        Initializes the TravelAgent.
        """
        self.chat_model = get_chat_model()
        self.conversation_history = [
            {"role": "system", "content": PREFERENCE_CAHT_PROMPT},
        ]
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "complete_perference_collection",
                    "description": "Completes the preference collection and returns a summary message.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            }
        ]

    def run(self, user_input: str) -> str:
        """
        Runs the travel agent for a single turn.

        Args:
            user_input: The user's input for this turn.

        Returns:
            The agent's response for this turn.
        """
        self.conversation_history.append({"role": "user", "content": user_input})

        response_message = self.chat_model.generate_response(
            self.conversation_history,
            tools=self.tools,
            tool_choice="auto",
        )

        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                if tool_call['name'] == "complete_perference_collection":
                    return self._complete_perference_collection()

        assistant_response = response_message.content
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_response}
        )
        return assistant_response

    def _complete_perference_collection(self):
        """
        Completes the preference collection and returns a summary message.

        Returns:
            str: A message containing the confirmation and the generated JSON.
        """
        print("\nThank you for sharing your travel preferences. I'm now making a summarize for you...")

        # Create a new conversation history for the JSON summary, starting with the specific prompt
        json_summary_conversation = [
            {"role": "system", "content": JSON_SUMMARY_PROMPT}
        ] + self.conversation_history[1:] # Exclude the initial SYSTEM_PROMPT

        response_message = self.chat_model.generate_response(
            json_summary_conversation, # Use the new conversation history with JSON_SUMMARY_PROMPT
            response_format={"type": "json_object"},
        )

        print(response_message)
        itinerary = json.loads(response_message.content)
        return f"Based on our conversation, here is the generated travel plan:\n{json.dumps(itinerary, indent=4)}"