import os
import logging
from google import genai

logger = logging.getLogger("LLMClient")

class LLMClientProvider:
    """
    Centralized Gemini client.
    Used by:
    - RouterAgent (Flash)
    - QueryPlannerAgent (Flash)
    - AnalystAgent (Pro)
    - CitationAgent (Pro)
    - FormatterAgent (Flash)
    """

    _client = None

    @classmethod
    def get_client(cls) -> genai.Client:
        if cls._client is None:
            api_key = os.getenv("API_KEY")
            logger.info("Initializing Gemini Client")
            cls._client = genai.Client(api_key=api_key)
        return cls._client

    @staticmethod
    def get_planner_model() -> str:
        return os.getenv("PLANNER_MODEL", "gemini-2.5-flash")

    @staticmethod
    def get_analyst_model() -> str:
        return os.getenv("ANALYST_MODEL", "gemini-2.5-pro")

    @staticmethod
    def get_router_model() -> str:
        return os.getenv("ROUTER_MODEL", "gemini-2.5-flash")

    @staticmethod
    def get_formatter_model() -> str:
        return os.getenv("FORMATTER_MODEL", "gemini-2.5-flash")
