#!/usr/bin/env python3
"""
API usage examples for ReasonIt.

This script demonstrates how to interact with the ReasonIt API server
using HTTP requests for reasoning and planning tasks.
"""

import asyncio
import json
import sys
from pathlib import Path

import aiohttp

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ReasonItAPIClient:
    """Simple client for ReasonIt API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def health_check(self):
        """Check API health."""
        async with self.session.get(f"{self.base_url}/health") as response:
            return await response.json()
    
    async def get_status(self):
        """Get system status."""
        async with self.session.get(f"{self.base_url}/status") as response:
            return await response.json()
    
    async def reason(self, query: str, **kwargs):
        """Submit reasoning request."""
        data = {"query": query, **kwargs}
        async with self.session.post(f"{self.base_url}/reason", json=data) as response:
            return await response.json()
    
    async def quick_reason(self, query: str, strategy: str = None):
        """Quick reasoning via GET endpoint."""
        params = {"q": query}
        if strategy:
            params["strategy"] = strategy
        
        async with self.session.get(f"{self.base_url}/quick", params=params) as response:
            return await response.json()
    
    async def create_plan(self, query: str, **kwargs):
        """Create execution plan."""
        data = {"query": query, **kwargs}
        async with self.session.post(f"{self.base_url}/plan", json=data) as response:
            return await response.json()
    
    async def get_strategies(self):
        """Get available strategies."""
        async with self.session.get(f"{self.base_url}/strategies") as response:
            return await response.json()
    
    async def get_metrics(self):
        """Get system metrics."""
        async with self.session.get(f"{self.base_url}/metrics") as response:
            return await response.json()
    
    async def batch_reason(self, requests: list):
        """Submit batch reasoning requests."""
        async with self.session.post(f"{self.base_url}/batch", json=requests) as response:
            return await response.json()


async def test_api_health():
    """Test API health and status."""
    
    print("🏥 API Health Check")
    print("=" * 50)
    
    async with ReasonItAPIClient() as client:
        try:
            # Health check
            health = await client.health_check()
            print(f"Health: {health['status']}")
            
            # System status
            status = await client.get_status()
            print(f"System initialized: {status['initialized']}")
            print(f"Available agents: {list(status['agents'].keys())}")
            
            # Available strategies
            strategies = await client.get_strategies()
            print(f"Available strategies: {len(strategies['strategies'])}")
            
            return True
            
        except Exception as e:
            print(f"❌ API not available: {e}")
            print("Make sure to start the API server first:")
            print("  python api_server.py")
            return False


async def test_basic_reasoning():
    """Test basic reasoning via API."""
    
    print("\n\n🧠 Basic Reasoning via API")
    print("=" * 50)
    
    async with ReasonItAPIClient() as client:
        # Simple math problem
        result = await client.reason(
            query="What is 25% of 400?",
            strategy="cot",
            confidence_threshold=0.8
        )
        
        print(f"Query: What is 25% of 400?")
        print(f"Answer: {result['final_answer']}")
        print(f"Confidence: {result['confidence_score']:.2%}")
        print(f"Cost: ${result['total_cost']:.4f}")
        print(f"Time: {result['total_time']:.2f}s")
        print(f"Strategy: {result['strategies_used']}")


async def test_quick_endpoint():
    """Test quick reasoning endpoint."""
    
    print("\n\n⚡ Quick Reasoning Endpoint")
    print("=" * 50)
    
    async with ReasonItAPIClient() as client:
        result = await client.quick_reason(
            query="Explain photosynthesis in one sentence",
            strategy="cot"
        )
        
        print(f"Quick answer: {result['final_answer']}")
        print(f"Confidence: {result['confidence_score']:.2%}")


async def test_different_strategies():
    """Test different reasoning strategies via API."""
    
    print("\n\n🎯 Different Strategies via API")
    print("=" * 50)
    
    query = "What are the pros and cons of remote work?"
    strategies = ["cot", "tot", "self_ask"]
    
    async with ReasonItAPIClient() as client:
        for strategy in strategies:
            print(f"\n{strategy.upper()} Strategy:")
            result = await client.reason(
                query=query,
                strategy=strategy,
                max_cost=0.08
            )
            
            print(f"  Answer length: {len(result['final_answer'])} chars")
            print(f"  Confidence: {result['confidence_score']:.2%}")
            print(f"  Cost: ${result['total_cost']:.4f}")


async def test_planning_api():
    """Test planning via API."""
    
    print("\n\n📋 Planning via API")
    print("=" * 50)
    
    async with ReasonItAPIClient() as client:
        plan = await client.create_plan(
            query="Create a marketing strategy for a new product launch",
            decomposition_strategy="hierarchical"
        )
        
        print(f"Plan ID: {plan['plan_id']}")
        print(f"Plan Name: {plan['name']}")
        print(f"Total Tasks: {plan['total_tasks']}")
        print(f"Estimated Cost: ${plan['estimated_cost']:.4f}")
        print(f"Estimated Time: {plan['estimated_time']:.2f}s")
        print(f"Success Rate: {plan['success_rate']:.2%}")


async def test_batch_processing():
    """Test batch processing via API."""
    
    print("\n\n📦 Batch Processing via API")
    print("=" * 50)
    
    requests = [
        {
            "query": "What is 2+2?",
            "strategy": "cot",
            "max_cost": 0.02
        },
        {
            "query": "What is the capital of France?",
            "strategy": "cot",
            "max_cost": 0.02
        },
        {
            "query": "Explain gravity in simple terms",
            "strategy": "cot",
            "max_cost": 0.05
        }
    ]
    
    async with ReasonItAPIClient() as client:
        results = await client.batch_reason(requests)
        
        print(f"Batch results: {len(results['results'])} responses")
        
        for i, result in enumerate(results['results']):
            if result['success']:
                answer = result['result']['final_answer']
                print(f"  {i+1}. {answer[:50]}{'...' if len(answer) > 50 else ''}")
            else:
                print(f"  {i+1}. Error: {result['error']}")


async def test_metrics():
    """Test metrics endpoint."""
    
    print("\n\n📊 System Metrics via API")
    print("=" * 50)
    
    async with ReasonItAPIClient() as client:
        metrics = await client.get_metrics()
        
        print(f"System Status: {metrics['system_status']}")
        
        if 'cost_summary' in metrics and metrics['cost_summary']:
            cost_summary = metrics['cost_summary']
            print(f"Total Cost: ${cost_summary.get('total_cost', 0):.4f}")
            print(f"Total Queries: {cost_summary.get('total_queries', 0)}")
        
        if 'planning_metrics' in metrics and metrics['planning_metrics']:
            planning = metrics['planning_metrics']
            print(f"Total Plans: {planning.get('total_plans', 0)}")
            print(f"Active Plans: {planning.get('active_plans', 0)}")


async def curl_examples():
    """Show equivalent curl commands."""
    
    print("\n\n🌐 Equivalent cURL Commands")
    print("=" * 50)
    
    print("Health check:")
    print("  curl http://localhost:8000/health")
    
    print("\nQuick reasoning:")
    print("  curl 'http://localhost:8000/quick?q=What+is+2+2?'")
    
    print("\nReasoning with POST:")
    print("  curl -X POST http://localhost:8000/reason \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"query\": \"Explain AI\", \"strategy\": \"cot\"}'")
    
    print("\nGet strategies:")
    print("  curl http://localhost:8000/strategies")
    
    print("\nSystem status:")
    print("  curl http://localhost:8000/status")


async def main():
    """Run API usage examples."""
    
    print("🚀 ReasonIt API Usage Examples")
    print("=" * 60)
    
    # Check if API is available
    api_available = await test_api_health()
    
    if not api_available:
        print("\n💡 To run these examples:")
        print("1. Start the API server: python api_server.py")
        print("2. Run this script again: python examples/api_usage.py")
        return
    
    try:
        # Run API tests
        await test_basic_reasoning()
        await test_quick_endpoint()
        await test_different_strategies()
        await test_planning_api()
        await test_batch_processing()
        await test_metrics()
        
        # Show curl examples
        await curl_examples()
        
        print("\n\n✅ All API examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running API examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())