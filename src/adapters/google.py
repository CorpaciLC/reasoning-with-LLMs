"""Google (Gemini) adapter implementation"""

from .provider import ProviderAdapter
import google.generativeai as genai
import json
from typing import List
from . import config  # Load environment variables
import os

import time
from typing import Optional

class GoogleAdapter(ProviderAdapter):
    def __init__(self):
        """Initialize Gemini adapter"""
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel('gemini-pro')
        self.last_call_time: Optional[float] = None
        self.min_delay = 1.0  # Minimum delay between calls in seconds

    def chat_completion(self, message: str) -> str:
        """Get completion from Gemini with rate limiting"""
        # Implement rate limiting
        if self.last_call_time is not None:
            elapsed = time.time() - self.last_call_time
            if elapsed < self.min_delay:
                time.sleep(self.min_delay - elapsed)
        
        try:
            self.last_call_time = time.time()
            response = self.model.generate_content(message)
            return response.text
        except Exception as e:
            print(f"Error in Gemini completion: {str(e)}")
            # Add exponential backoff for rate limits
            if 'quota' in str(e).lower() or '429' in str(e):
                time.sleep(self.min_delay * 2)
                self.min_delay *= 2
            return ""

    def extract_json_from_response(self, input_response: str) -> List[List[int]]:
        """Extract JSON from Gemini response"""
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