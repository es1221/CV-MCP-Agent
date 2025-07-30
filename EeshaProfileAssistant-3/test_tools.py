#!/usr/bin/env python3
"""
Test script to verify CV tools are working correctly
"""

import logging
from tools import CVTools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_tools():
    """Test all CV tools directly"""
    print("Testing CV Tools...\n")
    
    # Initialize tools
    tools = CVTools()
    
    # Test 1: List all repos
    print("=" * 50)
    print("TEST 1: List All Repositories")
    print("=" * 50)
    repos = tools.list_all_repos()
    print(repos)
    print()
    
    # Test 2: Get education
    print("=" * 50)
    print("TEST 2: Get Education")
    print("=" * 50)
    education = tools.get_education()
    print(education)
    print()
    
    # Test 3: Get skills
    print("=" * 50)
    print("TEST 3: Get Skills")
    print("=" * 50)
    skills = tools.get_skills()
    print(skills)
    print()
    
    # Test 4: Retrieve background with query
    print("=" * 50)
    print("TEST 4: Retrieve Background (query: 'Eesha Sondhi experience')")
    print("=" * 50)
    background = tools.retrieve_background("Eesha Sondhi experience")
    print(background)
    print()
    
    # Test 5: Get Butterfly Air info
    print("=" * 50)
    print("TEST 5: Get Butterfly Air Info")
    print("=" * 50)
    butterfly = tools.get_butterfly_air_info()
    print(butterfly)
    print()
    
    # Test 6: Check if repo_summaries.json is loaded
    print("=" * 50)
    print("TEST 6: Check repo_summaries")
    print("=" * 50)
    print(f"Number of repos loaded: {len(tools.repo_summaries)}")
    if tools.repo_summaries:
        print("First few repos:")
        for i, (name, _) in enumerate(list(tools.repo_summaries.items())[:3]):
            print(f"  - {name}")
    print()

if __name__ == "__main__":
    test_tools()