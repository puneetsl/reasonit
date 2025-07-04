#!/usr/bin/env python3
"""Test script to verify agent creation works."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that we can import the required modules."""
    try:
        print("Testing imports...")
        
        # Test model imports
        from models import SystemConfiguration, ReasoningRequest, ReasoningStrategy
        print("✅ Models imported successfully")
        
        # Test agent imports
        from agents.base_agent import BaseReasoningAgent
        print("✅ BaseReasoningAgent imported successfully")
        
        from agents.cot_agent import ChainOfThoughtAgent
        print("✅ ChainOfThoughtAgent imported successfully")
        
        from controllers.adaptive_controller import AdaptiveController
        print("✅ AdaptiveController imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_creation():
    """Test creating a simple agent."""
    try:
        print("\nTesting agent creation...")
        
        from models import SystemConfiguration
        from agents.cot_agent import ChainOfThoughtAgent
        
        # Create config
        config = SystemConfiguration()
        print("✅ Config created")
        
        # Try to create agent
        agent = ChainOfThoughtAgent(config=config)
        print("✅ ChainOfThoughtAgent created successfully")
        
        return agent
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_controller_creation():
    """Test creating the adaptive controller."""
    try:
        print("\nTesting controller creation...")
        
        from models import SystemConfiguration  
        from controllers.adaptive_controller import AdaptiveController
        
        # Create config
        config = SystemConfiguration()
        print("✅ Config created")
        
        # Try to create controller
        controller = AdaptiveController(config=config)
        print("✅ AdaptiveController created successfully")
        
        return controller
        
    except Exception as e:
        print(f"❌ Controller creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_simple_reasoning():
    """Test a simple reasoning call."""
    try:
        print("\nTesting simple reasoning...")
        
        from models import ReasoningRequest, ReasoningStrategy
        from controllers.adaptive_controller import AdaptiveController
        
        # Create controller
        controller = test_controller_creation()
        if not controller:
            print("❌ Cannot test reasoning without controller")
            return False
        
        # Create a simple request
        request = ReasoningRequest(
            query="What is 2 + 2?",
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            max_cost=0.01,
            max_time=30
        )
        print("✅ Request created")
        
        # Try reasoning
        result = await controller.reason(request)
        print(f"✅ Reasoning completed: {result.final_answer[:100]}...")
        print(f"   Confidence: {result.confidence_score:.2f}")
        print(f"   Cost: ${result.total_cost:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Reasoning failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("🧪 Testing Agent System")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        return False
    
    # Test agent creation
    agent = test_agent_creation()
    if not agent:
        return False
    
    # Test controller creation
    controller = test_controller_creation()
    if not controller:
        return False
    
    # Test simple reasoning
    success = await test_simple_reasoning()
    if not success:
        return False
    
    print("\n🎉 All tests passed!")
    return True

if __name__ == "__main__":
    asyncio.run(main())