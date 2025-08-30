"""
Configuration file for the Portfolio Management Application
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")  # Default to gpt-5-mini

# Validate configuration
if not OPENAI_API_KEY:
    print("⚠️  WARNING: OPENAI_API_KEY not found in environment variables.")
    print("   Please set your OpenAI API key in a .env file or environment variable.")
    print("   Get your API key from: https://platform.openai.com/api-keys")
    print("   Example .env file content:")
    print("   OPENAI_API_KEY=your_actual_api_key_here")
    print("   OPENAI_MODEL=gpt-5")
    print()

# Available models for easy switching - Updated with latest GPT-5 models
AVAILABLE_MODELS = {
    "gpt-5": "GPT-5 (latest and most capable - best for coding and agentic tasks)",
    "gpt-5-mini": "GPT-5 Mini (faster, cost-efficient for well-defined tasks)",
    "gpt-5-nano": "GPT-5 Nano (fastest, most cost-efficient for summarization and classification)",
    "gpt-5-chat": "GPT-5 Chat (used in ChatGPT with reasoning capabilities)",
    "gpt-4o": "GPT-4 Omni (previous generation - still very capable)",
    "gpt-4o-mini": "GPT-4 Omni Mini (faster, more affordable)",
    "gpt-4-turbo": "GPT-4 Turbo (previous generation - legacy compatibility)"
}

def get_model_info():
    """Get information about the current model configuration."""
    current_model = OPENAI_MODEL
    model_description = AVAILABLE_MODELS.get(current_model, "Unknown model")
    
    return {
        "current_model": current_model,
        "description": model_description,
        "available_models": list(AVAILABLE_MODELS.keys()),
        "recommended_model": "gpt-5"  # GPT-5 is now the recommended choice
    }

def update_model(new_model: str):
    """Update the model configuration."""
    global OPENAI_MODEL
    if new_model in AVAILABLE_MODELS:
        OPENAI_MODEL = new_model
        print(f"✅ Model updated to: {new_model}")
        print(f"📝 Description: {AVAILABLE_MODELS[new_model]}")
        return True
    else:
        print(f"❌ Invalid model: {new_model}")
        print(f"Available models: {', '.join(AVAILABLE_MODELS.keys())}")
        return False

def get_gpt5_variants():
    """Get information about GPT-5 variants specifically."""
    gpt5_models = {k: v for k, v in AVAILABLE_MODELS.items() if k.startswith('gpt-5')}
    return {
        "gpt5_variants": gpt5_models,
        "recommendations": {
            "gpt-5": "Best for complex portfolio analysis and coding tasks",
            "gpt-5-mini": "Great for standard portfolio operations with cost efficiency",
            "gpt-5-nano": "Perfect for quick portfolio summaries and classifications",
            "gpt-5-chat": "Excellent for conversational portfolio management"
        }
    }
