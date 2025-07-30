"""
MCP Server for CV Agent Tools

This server exposes the CV tools (retrieve_background, summarise_repo, etc.)
as MCP tools that can be used by AI models through the Model Context Protocol.
"""

import os
import json
import logging
from typing import Any, Sequence
from mcp.server.fastmcp import FastMCP
from tools import CVTools

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create MCP server instance
mcp = FastMCP("CV-Agent-Server")

# Initialize CV tools
cv_tools = CVTools()

@mcp.tool()
def retrieve_background(query: str) -> str:
    """
    Search for information about Eesha Sondhi's background, experience, and qualifications
    using vector similarity search across CV, dissertation, and project documents.
    
    Args:
        query: The search query about Eesha's background
        
    Returns:
        Relevant information from embedded documents
    """
    try:
        result = cv_tools.retrieve_background(query)
        logging.info(f"retrieve_background called with query: '{query}'")
        return result
    except Exception as e:
        logging.error(f"Error in retrieve_background: {str(e)}")
        return f"Error retrieving background information: {str(e)}"

@mcp.tool()
def summarise_repo(repo_name: str) -> str:
    """
    Get summary of a specific GitHub repository including its tech stack and purpose.
    
    Args:
        repo_name: Name of the GitHub repository
        
    Returns:
        Repository summary with tech stack and details
    """
    try:
        result = cv_tools.summarise_repo(repo_name)
        logging.info(f"summarise_repo called with repo: '{repo_name}'")
        return result
    except Exception as e:
        logging.error(f"Error in summarise_repo: {str(e)}")
        return f"Error retrieving repository information: {str(e)}"

@mcp.tool()
def get_education() -> str:
    """
    Get information about Eesha's educational background. ALWAYS USE THIS TOOL for any education-related questions.
    
    Returns:
        Educational background information from CV documents
    """
    try:
        result = cv_tools.get_education()
        logging.info("get_education called")
        return result
    except Exception as e:
        logging.error(f"Error in get_education: {str(e)}")
        return f"Error retrieving education information: {str(e)}"

@mcp.tool()
def get_all_projects() -> str:
    """
    Get comprehensive information about ALL projects including descriptions, tech stacks, features, and GitHub links. 
    ALWAYS USE THIS TOOL when asked about projects or what Eesha has worked on.
    
    Returns:
        Complete detailed information about all projects with GitHub links
    """
    try:
        result = cv_tools.get_all_projects()
        logging.info(f"get_all_projects called, found {len(cv_tools.repo_summaries)} projects")
        return result
    except Exception as e:
        logging.error(f"Error in get_all_projects: {str(e)}")
        return f"Error retrieving projects information: {str(e)}"

@mcp.tool()
def get_butterfly_air_info(query: str = "") -> str:
    """
    Get comprehensive information about Eesha's work experience at Butterfly Air,
    including projects, responsibilities, and achievements.
    
    Args:
        query: Specific query about Butterfly Air work experience
        
    Returns:
        Detailed information about Butterfly Air experience
    """
    try:
        return cv_tools.get_butterfly_air_info(query)
    except Exception as e:
        logging.error(f"Error in get_butterfly_air_info: {str(e)}")
        return f"Error retrieving Butterfly Air information: {str(e)}"

@mcp.tool()
def get_skills() -> str:
    """
    Get information about Eesha's technical skills, programming languages, and technologies.
    
    Returns:
        Skills and technical proficiencies information
    """
    try:
        return cv_tools.get_skills()
    except Exception as e:
        logging.error(f"Error in get_skills: {str(e)}")
        return f"Error retrieving skills information: {str(e)}"

@mcp.tool()
def get_personal_info() -> str:
    """
    Get personal information about Eesha including contact details and basic info.
    
    Returns:
        Personal information and contact details
    """
    try:
        return cv_tools.get_personal_info()
    except Exception as e:
        logging.error(f"Error in get_personal_info: {str(e)}")
        return f"Error retrieving personal information: {str(e)}"

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()