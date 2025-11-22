import os
import json
from openai import AzureOpenAI


class AzureChatModel:
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )

    def generate_response(
        self, messages, tools=None, tool_choice=None, response_format=None
    ):
        response = self.client.chat.completions.create(
            model=os.environ["AZURE_OPENAI_MODEL"],
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
        )
        return response.choices[0].message