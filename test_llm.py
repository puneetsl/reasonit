#!/usr/bin/env python3
"""
Simple test to verify OpenAI API connection and LLM wrapper functionality.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from models.openai_wrapper import OpenAIWrapper
from models import SystemConfiguration

async def test_direct_llm():
    """Test direct LLM wrapper call."""
    
    print("Testing direct OpenAI wrapper...")
    
    config = SystemConfiguration()
    wrapper = OpenAIWrapper(model_name="gpt-4o-mini", config=config)
    
    try:
        response = await wrapper.generate(
            prompt="What is 2+2? Answer briefly.",
            session_id="test_session"
        )
        
        print(f"✅ SUCCESS: Got response: '{response}'")
        print(f"📊 Usage: {wrapper.get_usage_metrics()}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_cot_agent():
    """Test Chain of Thought agent directly."""
    
    print("\nTesting CoT agent...")
    
    from agents import ChainOfThoughtAgent
    from models import ReasoningRequest, ReasoningStrategy, ContextVariant
    
    try:
        agent = ChainOfThoughtAgent(config=SystemConfiguration())
        
        request = ReasoningRequest(
            query="What is 2+2?",
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            context_variant=ContextVariant.STANDARD,
            session_id="test_session"
        )
        
        result = await agent.reason(request)
        
        print(f"✅ SUCCESS: Agent response: '{result.final_answer}'")
        print(f"📊 Confidence: {result.confidence_score}, Cost: ${result.total_cost}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    
    print("🧪 Testing ReasonIt LLM Integration")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No OPENAI_API_KEY found!")
        return
    
    print(f"🔑 API Key found: {api_key[:10]}...")
    
    # Test direct wrapper
    wrapper_ok = await test_direct_llm()
    
    # Test agent
    agent_ok = await test_cot_agent()
    
    if wrapper_ok and agent_ok:
        print("\n🎉 All tests passed! LLM integration is working.")
    else:
        print("\n😞 Some tests failed. Check the errors above.")

if __name__ == "__main__":
    asyncio.run(main())