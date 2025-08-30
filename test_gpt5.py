#!/usr/bin/env python3
"""
Test script for GPT-5 integration with the Portfolio Management Application
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_gpt5_integration():
    """Test the GPT-5 integration."""
    
    print("🧪 Testing GPT-5 Integration")
    print("=" * 50)
    
    # Check if OpenAI API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("   Please set your OpenAI API key in a .env file")
        print("   Example: OPENAI_API_KEY=your_actual_api_key_here")
        return False
    
    print("✅ OpenAI API key found")
    
    # Test importing the LangGraph model
    try:
        from langgraph_model import PortfolioAgent, chat_with_portfolio_agent
        print("✅ LangGraph model imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import LangGraph model: {e}")
        print("   Please install dependencies: pip install -r requirements.txt")
        return False
    
    # Test creating a PortfolioAgent with GPT-5
    try:
        agent = PortfolioAgent(model_name="gpt-5")
        print("✅ PortfolioAgent created with GPT-5")
        print(f"   Model: {agent.model_name}")
    except Exception as e:
        print(f"❌ Failed to create PortfolioAgent: {e}")
        return False
    
    # Test the chat function
    try:
        test_portfolio = {"AAPL": 30, "GOOGL": 25}
        response = chat_with_portfolio_agent("What's in my portfolio?", test_portfolio)
        print("✅ Chat function working")
        print(f"   Response: {response[:100]}...")
    except Exception as e:
        print(f"❌ Chat function failed: {e}")
        return False
    
    print("\n🎉 All tests passed! GPT-5 integration is working correctly.")
    return True

def test_model_switching():
    """Test model switching functionality."""
    
    print("\n🔄 Testing Model Switching")
    print("=" * 50)
    
    try:
        from config import get_model_info, update_model, get_gpt5_variants
        
        # Get current model info
        model_info = get_model_info()
        print(f"✅ Current model: {model_info['current_model']}")
        print(f"   Description: {model_info['description']}")
        
        # Get GPT-5 variants
        gpt5_info = get_gpt5_variants()
        print(f"\n🆕 Available GPT-5 models:")
        for model, desc in gpt5_info['gpt5_variants'].items():
            print(f"   • {model}: {desc}")
        
        # Test switching to GPT-5-mini
        print(f"\n🔄 Testing switch to GPT-5-mini...")
        success = update_model("gpt-5-mini")
        if success:
            print("✅ Successfully switched to GPT-5-mini")
            
            # Switch back to GPT-5
            print("🔄 Switching back to GPT-5...")
            success = update_model("gpt-5")
            if success:
                print("✅ Successfully switched back to GPT-5")
            else:
                print("❌ Failed to switch back to GPT-5")
                return False
        else:
            print("❌ Failed to switch to GPT-5-mini")
            return False
            
    except Exception as e:
        print(f"❌ Model switching test failed: {e}")
        return False
    
    print("\n🎉 Model switching tests passed!")
    return True

if __name__ == "__main__":
    print("🚀 GPT-5 Integration Test Suite")
    print("=" * 60)
    
    # Run tests
    test1_passed = test_gpt5_integration()
    test2_passed = test_model_switching()
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ GPT-5 integration is working correctly")
        print("✅ Model switching is functional")
        print("\n🚀 Your Portfolio Management Application is ready to use GPT-5!")
    else:
        print("❌ Some tests failed")
        print("   Please check the error messages above")
        sys.exit(1)
