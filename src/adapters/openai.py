"""OpenAI (GPT) adapter implementation"""

from .provider import ProviderAdapter
import os
from openai import OpenAI
import json
from typing import List
from . import config  # Load environment variables

class OpenAIAdapter(ProviderAdapter):
    def __init__(self):
        """Initialize GPT adapter"""
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = "gpt-4"

    def chat_completion(self, message: str) -> str:
        """Get completion from GPT"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": message}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in GPT completion: {str(e)}")
            return ""

    def extract_json_from_response(self, input_response: str) -> List[List[int]]:
        """Extract JSON from GPT response"""
        try:
            # Find JSON-like content between triple backticks
            json_str = input_response.split("```json")[1].split("```")[0]
            return json.loads(json_str)
        except:
            try:
                # Try direct JSON parsing if no backticks
                return json.loads(input_response)
            except:
                return []