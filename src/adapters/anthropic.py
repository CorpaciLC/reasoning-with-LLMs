"""Anthropic (Claude) adapter implementation"""

from .provider import ProviderAdapter
import os
import anthropic
import json
from typing import List
from . import config  # Load environment variables

class AnthropicAdapter(ProviderAdapter):
    def __init__(self):
        """Initialize Claude adapter"""
        self.client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )
        self.model = "claude-3-sonnet-20240229"

    def chat_completion(self, message: str) -> str:
        """Get completion from Claude"""
        try:
            response = self.client.messages.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": message
                }],
                max_tokens=1024,
            )
            return response.content[0].text
        except Exception as e:
            print(f"Error in Claude completion: {str(e)}")
            return ""

    def extract_json_from_response(self, input_response: str) -> List[List[int]]:
        """Extract JSON from Claude response"""
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