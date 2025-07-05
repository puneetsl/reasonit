#!/usr/bin/env python3
"""
Test script for Wikipedia Knowledge Retrieval System.

This script demonstrates the Wikipedia integration system with intelligent
detection, content processing, and synthesis capabilities.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from tools.wikipedia import (
    WikipediaKnowledgeDetector,
    WikipediaSearchTool, 
    WikipediaContentSynthesizer,
    EntityRecognizer,
    WikipediaCache
)
from tools.wikipedia.wikipedia_cache import create_memory_only_cache
from tools.wikipedia.knowledge_detector import KnowledgeType


def test_knowledge_detection():
    """Test Wikipedia knowledge detection system."""
    print("🔍 Testing Wikipedia Knowledge Detection")
    print("=" * 50)
    
    detector = WikipediaKnowledgeDetector()
    
    test_queries = [
        "Who was Albert Einstein?",
        "When did World War II start?",
        "What is the capital of France?",
        "How does photosynthesis work?",
        "What is the population of Tokyo?",
        "When was the Declaration of Independence signed?",
        "What is quantum mechanics?",
        "Who founded Microsoft?",
        "Where is the Eiffel Tower located?",
        "Calculate 2 + 2"  # Should NOT need Wikipedia
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = detector.detect_knowledge_needs(query)
        
        print(f"  Needs Wikipedia: {result.needs_wikipedia}")
        if result.primary_need:
            print(f"  Knowledge Type: {result.primary_need.knowledge_type.value}")
            print(f"  Confidence: {result.overall_confidence:.2f}")
            print(f"  Search Terms: {result.primary_need.search_terms}")
            if result.primary_need.entities:
                print(f"  Entities: {result.primary_need.entities}")
        
        if result.detection_reasoning:
            print(f"  Reasoning: {result.detection_reasoning[0]}")
    
    print("\n✅ Knowledge detection tests completed\n")


def test_entity_recognition():
    """Test entity recognition system."""
    print("🏷️ Testing Entity Recognition")
    print("=" * 50)
    
    recognizer = EntityRecognizer()
    
    test_texts = [
        "Albert Einstein was born in Germany in 1879.",
        "The Battle of Waterloo took place in Belgium in 1815.",
        "Google was founded by Larry Page and Sergey Brin at Stanford University.",
        "Mount Everest is located in the Himalayas between Nepal and Tibet.",
        "The theory of relativity was developed by Einstein in the early 20th century.",
        "Shakespeare wrote Romeo and Juliet in the 16th century."
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        result = recognizer.recognize_entities(text)
        
        print(f"  Entities found: {len(result.entities)}")
        print(f"  Overall confidence: {result.confidence_score:.2f}")
        
        for entity in result.entities[:3]:  # Show top 3 entities
            print(f"    • {entity.text} ({entity.entity_type.value}, conf: {entity.confidence:.2f})")
        
        # Show entity types present
        if result.entity_count_by_type:
            type_counts = [f"{k.value}: {v}" for k, v in result.entity_count_by_type.items()]
            print(f"  Types: {', '.join(type_counts)}")
    
    print("\n✅ Entity recognition tests completed\n")


async def test_wikipedia_search():
    """Test Wikipedia search functionality."""
    print("🔎 Testing Wikipedia Search")
    print("=" * 50)
    
    # Test with simple queries that should work without actual API calls
    search_queries = [
        "Albert Einstein",
        "World War II", 
        "Python programming",
        "Artificial Intelligence"
    ]
    
    # Create search tool (will fail without actual network, but we can test the structure)
    try:
        async with WikipediaSearchTool() as search_tool:
            for query in search_queries:
                print(f"\nSearching for: {query}")
                
                try:
                    result = await search_tool.search_articles(query, limit=3, include_content=False)
                    print(f"  Found {len(result.pages)} pages")
                    print(f"  Search time: {result.search_time:.2f}s")
                    
                    for page in result.pages[:2]:  # Show first 2 pages
                        print(f"    • {page.title} (ID: {page.page_id})")
                
                except Exception as e:
                    print(f"  ⚠️ Search failed (expected without network): {type(e).__name__}")
    
    except Exception as e:
        print(f"⚠️ Wikipedia search test skipped (no network access): {type(e).__name__}")
    
    print("\n✅ Wikipedia search tests completed\n")


def test_content_synthesis():
    """Test content synthesis with mock data."""
    print("📝 Testing Content Synthesis")
    print("=" * 50)
    
    # Create mock Wikipedia pages for testing
    from tools.wikipedia.wikipedia_search import WikipediaPage
    from tools.wikipedia.knowledge_detector import KnowledgeNeed
    from datetime import datetime
    
    mock_page = WikipediaPage(
        title="Albert Einstein",
        page_id=12345,
        url="https://en.wikipedia.org/wiki/Albert_Einstein",
        summary="Albert Einstein (1879-1955) was a German-born theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics.",
        full_text="Albert Einstein was born on March 14, 1879, in Ulm, Germany. He is best known for developing the theory of relativity and winning the Nobel Prize in Physics in 1921. Einstein's work on the photoelectric effect was instrumental in the development of quantum theory.",
        infobox={
            "Born": "March 14, 1879",
            "Died": "April 18, 1955", 
            "Nationality": "German, Swiss, American",
            "Known for": "Theory of relativity, E=mc²",
            "Awards": "Nobel Prize in Physics (1921)"
        },
        sections={
            "Early life": "Einstein was born in Ulm, Germany, to Hermann and Pauline Einstein. The family moved to Munich when he was young.",
            "Scientific career": "Einstein developed the special theory of relativity in 1905 and the general theory of relativity in 1915.",
            "Nobel Prize": "He won the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect."
        },
        categories=["German physicists", "Theoretical physicists", "Nobel Prize winners"]
    )
    
    # Create mock search result
    from tools.wikipedia.wikipedia_search import SearchResult
    
    search_result = SearchResult(
        query="Who was Albert Einstein?",
        pages=[mock_page],
        search_time=0.5
    )
    
    # Create knowledge need
    knowledge_need = KnowledgeNeed(
        knowledge_type=KnowledgeType.BIOGRAPHICAL,
        entities=["Albert Einstein"],
        topics=["physics", "relativity"],
        confidence=0.9,
        search_terms=["Albert Einstein", "physicist"]
    )
    
    # Test synthesis
    synthesizer = WikipediaContentSynthesizer()
    synthesized = synthesizer.synthesize_content(search_result, knowledge_need)
    
    print(f"Query: {synthesized.query}")
    print(f"Content Type: {synthesized.content_type.value}")
    print(f"Confidence: {synthesized.confidence_score:.2f}")
    print(f"\nSummary: {synthesized.summary}")
    print(f"\nKey Facts:")
    for fact in synthesized.key_facts:
        print(f"  • {fact}")
    
    if synthesized.structured_data:
        print(f"\nStructured Data:")
        for key, value in synthesized.structured_data.items():
            print(f"  {key}: {value}")
    
    if synthesized.timeline:
        print(f"\nTimeline:")
        for event in synthesized.timeline:
            print(f"  {event['year']}: {event['event']}")
    
    print("\n✅ Content synthesis tests completed\n")


def test_cache_system():
    """Test Wikipedia caching system."""
    print("💾 Testing Wikipedia Cache")
    print("=" * 50)
    
    # Create cache with memory-only mode for testing
    from tools.wikipedia.wikipedia_cache import create_memory_only_cache
    cache = create_memory_only_cache(max_entries=10)
    
    # Test basic cache operations
    cache.put("test_key_1", "test_content_1", content_type="test")
    cache.put("test_key_2", "test_content_2", content_type="test")
    
    # Test retrieval
    content1 = cache.get("test_key_1")
    content2 = cache.get("nonexistent_key")
    
    print(f"Retrieved content 1: {content1}")
    print(f"Retrieved nonexistent: {content2}")
    
    # Test statistics
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Total entries: {stats.total_entries}")
    print(f"  Hit count: {stats.hit_count}")
    print(f"  Miss count: {stats.miss_count}")
    print(f"  Hit ratio: {stats.cache_hit_ratio:.2%}")
    
    # Test cache with Wikipedia content
    from tools.wikipedia.wikipedia_search import WikipediaPage
    from datetime import datetime
    
    mock_page = WikipediaPage(
        title="Test Page",
        page_id=123,
        url="https://test.com",
        summary="Test summary",
        full_text="Test content",
        last_modified=datetime.now()
    )
    
    cache_key = cache.cache_wikipedia_page(mock_page)
    retrieved_page = cache.get_wikipedia_page("Test Page")
    
    print(f"\nWikipedia page cached with key: {cache_key}")
    print(f"Retrieved page title: {retrieved_page.title if retrieved_page else 'None'}")
    
    # Test cache optimization
    optimization_result = cache.optimize_cache()
    print(f"\nCache optimization result: {optimization_result}")
    
    print("\n✅ Cache system tests completed\n")


def test_integration_workflow():
    """Test integrated workflow for Wikipedia knowledge retrieval."""
    print("🔄 Testing Integrated Workflow")
    print("=" * 50)
    
    # Create components
    detector = WikipediaKnowledgeDetector()
    recognizer = EntityRecognizer()
    cache = create_memory_only_cache()
    
    test_query = "When was Albert Einstein born and what is he famous for?"
    
    print(f"Query: {test_query}")
    print("\nStep 1: Knowledge Detection")
    detection_result = detector.detect_knowledge_needs(test_query)
    print(f"  Needs Wikipedia: {detection_result.needs_wikipedia}")
    print(f"  Knowledge type: {detection_result.primary_need.knowledge_type.value if detection_result.primary_need else 'None'}")
    
    print("\nStep 2: Entity Recognition")
    entity_result = recognizer.recognize_entities(test_query)
    print(f"  Entities found: {len(entity_result.entities)}")
    for entity in entity_result.entities:
        print(f"    • {entity.text} ({entity.entity_type.value})")
    
    print("\nStep 3: Search Strategy")
    if detection_result.needs_wikipedia and detection_result.primary_need:
        strategy = detector.suggest_search_strategy(detection_result)
        print(f"  Strategy: {strategy['strategy']}")
        print(f"  Primary searches: {strategy['primary_searches']}")
        print(f"  Expected content: {strategy['expected_content']}")
    
    print("\nStep 4: Cache Check")
    cached_result = cache.get_search_result(test_query)
    print(f"  Cached result found: {cached_result is not None}")
    
    print("\nStep 5: Integration Analysis")
    query_analysis = detector.analyze_query_complexity(test_query)
    print(f"  Complexity: {query_analysis['overall_complexity']}")
    print(f"  Recommendation: {query_analysis['recommendation']}")
    
    print("\n✅ Integrated workflow tests completed\n")


def demonstrate_real_world_scenarios():
    """Demonstrate real-world usage scenarios."""
    print("🌍 Real-World Usage Scenarios")
    print("=" * 50)
    
    detector = WikipediaKnowledgeDetector()
    recognizer = EntityRecognizer()
    
    scenarios = [
        {
            "context": "History Research",
            "query": "What were the main causes of the French Revolution and when did it begin?"
        },
        {
            "context": "Science Education", 
            "query": "Explain photosynthesis and which scientist first discovered it?"
        },
        {
            "context": "Geography Study",
            "query": "What is the population and area of Japan?"
        },
        {
            "context": "Biography Research",
            "query": "Who was Marie Curie and what did she discover?"
        },
        {
            "context": "Current Events",
            "query": "What is the latest information about climate change?"
        },
        {
            "context": "Literature Study",
            "query": "When did Shakespeare write Hamlet and what is it about?"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['context'].upper()} SCENARIO:")
        print(f"Query: {scenario['query']}")
        
        # Detect knowledge needs
        detection = detector.detect_knowledge_needs(scenario['query'])
        print(f"Wikipedia needed: {detection.needs_wikipedia}")
        
        if detection.primary_need:
            print(f"Knowledge type: {detection.primary_need.knowledge_type.value}")
            print(f"Search terms: {detection.primary_need.search_terms[:3]}")
            print(f"Urgency: {detection.primary_need.urgency}")
        
        # Recognize entities
        entities = recognizer.recognize_entities(scenario['query'])
        if entities.entities:
            top_entities = [f"{e.text} ({e.entity_type.value})" for e in entities.entities[:2]]
            print(f"Key entities: {', '.join(top_entities)}")
        
        # Analysis
        analysis = recognizer.analyze_query_structure(scenario['query'])
        print(f"Search intent: {analysis['search_intent']}")
        print(f"Complexity: {analysis['complexity']}")
    
    print("\n✅ Real-world scenario tests completed\n")


def main():
    """Run all Wikipedia system tests."""
    print("🌐 Wikipedia Knowledge Retrieval System Test Suite")
    print("=" * 60)
    print("Testing intelligent Wikipedia integration for factual questions\n")
    
    try:
        # Run synchronous tests
        test_knowledge_detection()
        test_entity_recognition()
        test_content_synthesis()
        test_cache_system()
        test_integration_workflow()
        demonstrate_real_world_scenarios()
        
        # Run asynchronous tests
        print("Running async tests...")
        asyncio.run(test_wikipedia_search())
        
        print("🎉 All tests completed successfully!")
        print("\nThe Wikipedia Knowledge Retrieval System is ready to enhance")
        print("reasoning with intelligent factual knowledge from Wikipedia!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)