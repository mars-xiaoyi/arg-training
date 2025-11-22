import os
from langchain_google_vertexai import ChatVertexAI


class GoogleChatModel:
    def __init__(self):
        self.model = ChatVertexAI(
            model_name=os.environ["GOOGLE_CHAT_MODEL"], 
            project=os.environ["GOOGLE_CLOUD_PROJECT"], 
            location=os.environ["GOOGLE_CLOUD_LOCATION"],
            temperature=0.7
        )

    def generate_response(
        self, messages, tools=None, tool_choice=None, response_format=None
    ):
        # LangChain uses a different message format, so we need to convert it
        converted_messages = []
        for msg in messages:
            if msg["role"] == "system":
                converted_messages.append(("system", msg["content"]))
            elif msg["role"] == "user":
                converted_messages.append(("human", msg["content"]))
            elif msg["role"] == "assistant":
                converted_messages.append(("ai", msg["content"]))

        if tools:
            self.model = self.model.bind_tools(tools)

        # Pass response_format to the invoke method if it's provided
        if response_format:
            response = self.model.invoke(converted_messages, response_format=response_format)
        else:
            response = self.model.invoke(converted_messages)
        return response