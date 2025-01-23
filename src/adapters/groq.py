"""Groq adapter implementation"""

from .provider import ProviderAdapter
from groq import Groq
import json
from typing import List
from . import config  # Load environment variables
import os

class GroqAdapter(ProviderAdapter):
    def __init__(self):
        """Initialize Groq adapter"""
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = "mixtral-8x7b-32768"  # or other available model

    def chat_completion(self, message: str) -> str:
        """Get completion from Groq"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": message
                }]
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error in Groq completion: {str(e)}")
            return ""

    def extract_json_from_response(self, input_response: str) -> List[List[int]]:
        """Extract JSON from Groq response"""
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