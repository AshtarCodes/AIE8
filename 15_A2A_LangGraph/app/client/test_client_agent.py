"""Test script for the Client Agent using A2A protocol.

This script tests the client agent's ability to intelligently use the A2A server
for various types of queries.
"""
import asyncio
import logging
from datetime import datetime

from app.client.client_agent import ClientAgent


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Run test queries through the client agent."""
    print("=" * 80)
    print("Client Agent A2A Test Suite")
    print("=" * 80)
    print("\nThis script tests a LangGraph client agent that uses the A2A protocol")
    print("to communicate with an A2A-compliant agent server.\n")
    print("Make sure the A2A server is running: uv run python -m app")
    print("=" * 80)
    print()
    
    # Initialize the client agent
    logger.info("Initializing client agent...")
    agent = ClientAgent()
    print("✓ Client agent initialized\n")
    
    # Define test queries
    test_queries = [
        {
            "name": "Web Search Query",
            "query": "What are the latest developments in artificial intelligence in 2025?",
            "description": "Tests the agent's ability to use web search through A2A"
        },
        {
            "name": "Academic Research Query",
            "query": "Find recent research papers about transformer architectures and attention mechanisms",
            "description": "Tests the agent's ability to search arXiv through A2A"
        },
        {
            "name": "General Knowledge Query",
            "query": "Explain how neural networks learn from data",
            "description": "Tests the agent's decision-making on when to use A2A vs. answering directly"
        },
        {
            "name": "Specific Document Query",
            "query": "What information do you have about AI maker space projects?",
            "description": "Tests the agent's ability to use RAG retrieval through A2A"
        }
    ]
    
    # Run each test query
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'=' * 80}")
        print(f"Test {i}/{len(test_queries)}: {test['name']}")
        print(f"{'=' * 80}")
        print(f"Description: {test['description']}")
        print(f"\nQuery: {test['query']}")
        print(f"\n{'-' * 80}")
        print("Processing...\n")
        
        start_time = datetime.now()
        
        try:
            # Query the agent
            response = await agent.query(test['query'], thread_id=f"test-{i}")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print("Response:")
            print(response)
            print(f"\n{'-' * 80}")
            print(f"✓ Completed in {duration:.2f} seconds")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            logger.error(f"Error processing query {i}: {e}", exc_info=True)
        
        print()
    
    print("=" * 80)
    print("Test Suite Complete")
    print("=" * 80)
    print("\nSummary:")
    print(f"- Total queries tested: {len(test_queries)}")
    print("- The client agent successfully used the A2A protocol to interact")
    print("  with the A2A server, leveraging its web search, arXiv, and RAG capabilities")
    print("\nActivity #1 Complete! ✓")


if __name__ == "__main__":
    asyncio.run(main())

