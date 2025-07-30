#!/usr/bin/env python3
"""
Test script to verify MCP Agent is working correctly and calling tools
"""

import logging
from mcp_client import MCPAgent

# Configure logging to see tool calls
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_mcp_agent():
    """Test MCP Agent with various queries"""
    print("Testing MCP Agent...\n")
    
    # Initialize agent
    agent = MCPAgent()
    
    # Test queries
    test_queries = [
        "What projects has Eesha worked on?",
        "What is Eesha's education?",
        "What are Eesha's technical skills?",
        "Tell me about Eesha's work at Butterfly Air",
        "What is Eesha's experience?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print("=" * 70)
        print(f"TEST {i}: {query}")
        print("=" * 70)
        
        try:
            response = agent.ask(query)
            print(f"Response:\n{response}\n")
        except Exception as e:
            print(f"Error: {str(e)}\n")

if __name__ == "__main__":
    test_mcp_agent()