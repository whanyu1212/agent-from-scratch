import requests
import json
import os
import re
from loguru import logger
from dotenv import load_dotenv
from typing import Dict

load_dotenv()


class GeminiModel:
    def __init__(self, model_name: str, system_prompt: str, temperature: float = 1.0):
        """Initializes a Gemini model instance.
        HTTP POST → Gemini API → Response → Parsed Result

        Args:
            model_name (str): the model name available in the Gemini API.
            system_prompt (str): the system prompt to prepend to the user's input.
            temperature (float, optional): parameter that controls the randomness of
            response. Defaults to 1.0.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        self.headers = {"Content-Type": "application/json"}

    def generate_text(self, prompt: str) -> Dict:
        """Generates text using the Gemini model via the API.

        Args:
           prompt: The user's input prompt.

        Returns:
             response: Response from the model in dict format.
        """
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": self.system_prompt
                        },  # Prepended system prompt (if any)
                        {"text": prompt},
                    ]
                }
            ]
        }
        logger.info(f"Payload sent to Gemini model: {payload}\n")

        response_dict = requests.post(
            self.model_endpoint, headers=self.headers, data=json.dumps(payload)
        )

        try:
            response_dict.raise_for_status()  # Raise an exception for bad responses (4xx or 5xx)

            response_json = response_dict.json()

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error occurred: {e}")
            response = {"response": "An error occurred. Please try again."}
            return response

        try:
            text_response = response_json["candidates"][0]["content"]["parts"][0][
                "text"
            ]
            # Remove backticks and newlines and any surrounding text
            cleaned_response = re.sub(
                r"```(?:json)?\n?(.*?)\n?```", r"\1", text_response, flags=re.DOTALL
            ).strip()
            response = json.loads(cleaned_response)
        except (KeyError, json.JSONDecodeError) as e:
            response = {"response": text_response}

        return response
