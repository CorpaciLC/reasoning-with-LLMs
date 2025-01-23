"""Configuration loader for LLM adapters"""
import os
from pathlib import Path
from dotenv import load_dotenv

def load_env():
    """Load environment variables from .env file"""
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)

# Load environment variables
load_env()