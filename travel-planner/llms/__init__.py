from config import CHAT_MODEL_PROVIDER
from dotenv import load_dotenv
import os

load_dotenv() # Load environment variables from .env file


def get_chat_model():
    if CHAT_MODEL_PROVIDER == "google":
        from .google import GoogleChatModel

        return GoogleChatModel()
    elif CHAT_MODEL_PROVIDER == "azure":
        from .azure import AzureChatModel

        return AzureChatModel()
    else:
        raise ValueError(f"Unknown chat model provider: {CHAT_MODEL_PROVIDER}")