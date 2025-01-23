"""DeepSeek adapter implementation"""

from .provider import ProviderAdapter
import requests
import json
from typing import List
from . import config  # Load environment variables
import os

class DeepSeekAdapter(ProviderAdapter):
    def __init__(self):
        """Initialize DeepSeek adapter"""
        self.api_key = os.environ["DeepSeek_API_KEY"]
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.model = "deepseek-chat"

    def chat_completion(self, message: str) -> str:
        """Get completion from DeepSeek"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": message}]
            }
            
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error in DeepSeek completion: {str(e)}")
            return ""

    def extract_json_from_response(self, input_response: str) -> List[List[int]]:
        """Extract JSON from DeepSeek response"""
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