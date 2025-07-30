"""
MCP Client for OpenAI Integration with Persistent Server

This client maintains a persistent connection to the MCP server to avoid
reinitializing embeddings on every tool call.
"""

import os
import asyncio
import json
import logging
import subprocess
import time
from typing import Dict, Any, List, Optional
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class PersistentMCPServer:
    """Manages a persistent MCP server process"""
    
    def __init__(self):
        self.process = None
        self.start_server()
    
    def start_server(self):
        """Start the MCP server as a subprocess"""
        try:
            # Start the MCP server process
            self.process = subprocess.Popen(
                ["python", "mcp_server.py"],
                env=os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Give the server time to start
            time.sleep(2)
            
            logging.info("Started persistent MCP server")
            
        except Exception as e:
            logging.error(f"Error starting MCP server: {str(e)}")
            raise
    
    def stop_server(self):
        """Stop the MCP server process"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            logging.info("Stopped MCP server")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.stop_server()

class MCPOpenAIClient:
    def __init__(self):
        """Initialize the MCP-OpenAI client with persistent server"""
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.tools = []
        self.server_params = StdioServerParameters(
            command="python",
            args=["mcp_server.py"],
            env=os.environ.copy()
        )
        
        # Initialize tools on creation
        asyncio.run(self._initialize_tools())
        
    async def _initialize_tools(self):
        """Initialize and cache the available tools"""
        try:
            async with stdio_client(self.server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    
                    # Get available tools from MCP server
                    tools_result = await session.list_tools()
                    
                    # Convert MCP tools to OpenAI function format
                    self.tools = []
                    for tool in tools_result.tools:
                        openai_tool = {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.inputSchema
                            }
                        }
                        self.tools.append(openai_tool)
                    
                    logging.info(f"Initialized MCP client with {len(self.tools)} tools")
                    
        except Exception as e:
            logging.error(f"Error initializing MCP client: {str(e)}")
            raise
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool on the MCP server"""
        try:
            # Reuse the same server params - server is already running
            async with stdio_client(self.server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    
                    # Execute the tool
                    result = await session.call_tool(tool_name, arguments=arguments)
                    
                    # Extract the text response
                    if result.content and len(result.content) > 0:
                        content_item = result.content[0]
                        if hasattr(content_item, 'text'):
                            return content_item.text
                        else:
                            return str(content_item)
                    else:
                        return "No response from tool"
                        
        except Exception as e:
            logging.error(f"Error executing tool {tool_name}: {str(e)}")
            return f"Error executing tool: {str(e)}"
    
    def _should_force_tool(self, question: str) -> str:
        """Determine if we should force a specific tool based on the question"""
        q_lower = question.lower()
        
        # Map keywords to tools
        if any(word in q_lower for word in ['project', 'projects', 'worked on', 'repository', 'repos']):
            return "get_all_projects"
        elif any(word in q_lower for word in ['education', 'degree', 'university', 'study', 'studied']):
            return "get_education"
        elif any(word in q_lower for word in ['skill', 'technology', 'programming', 'language', 'tech stack']):
            return "get_skills"
        elif any(word in q_lower for word in ['butterfly', 'butterfly air', 'work experience']):
            return "get_butterfly_air_info"
        elif any(word in q_lower for word in ['contact', 'email', 'phone', 'personal']):
            return "get_personal_info"
        
        return None
    
    async def ask(self, question: str) -> str:
        """
        Ask a question using OpenAI with MCP tools available
        
        Args:
            question: The user's question
            
        Returns:
            AI response incorporating tool results if needed
        """
        try:
            # Check if we should force a specific tool
            forced_tool = self._should_force_tool(question)
            
            # Create the conversation with system message
            messages: List[Dict[str, Any]] = [
                {
                    "role": "system", 
                    "content": """You are a professional CV agent for Eesha Sondhi. Your role is to answer questions about her background, experience, education, and projects based on available information. Eesha's focus is on applied AI and ML - make sure your responses reflect this.

IMPORTANT: You MUST use the appropriate tools to retrieve information before answering ANY question about Eesha. Do not make up information.

Tool Usage Guidelines:
- For questions about projects or repositories: ALWAYS use get_all_projects() to see all projects with full details and GitHub links
- For specific project details: Use summarise_repo() with the exact repository name for focused information
- For education questions: ALWAYS use get_education() 
- For skills or technical abilities: ALWAYS use get_skills()
- For work experience at Butterfly Air: ALWAYS use get_butterfly_air_info()
- For any other background questions: Use retrieve_background() with a relevant query
- For personal/contact information: Use get_personal_info()

When users ask about projects, the get_all_projects() tool will automatically include GitHub repository links and comprehensive details.

NEVER answer without first calling the relevant tool(s). If a tool returns no information, acknowledge that the information is not available rather than making something up."""
                },
                {"role": "user", "content": question}
            ]
            
            # Create initial OpenAI completion with tools
            tool_choice = "auto"
            if forced_tool:
                # Force specific tool usage
                tool_choice = {"type": "function", "function": {"name": forced_tool}}
                logging.info(f"Forcing tool usage: {forced_tool}")
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=self.tools,
                tool_choice=tool_choice,
                temperature=0.7
            )
            
            # Check if OpenAI wants to use tools
            if response.choices[0].message.tool_calls:
                # Add assistant's message with tool calls to conversation
                assistant_message = {
                    "role": "assistant",
                    "content": response.choices[0].message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in response.choices[0].message.tool_calls
                    ]
                }
                messages.append(assistant_message)
                
                # Execute each tool call
                for tool_call in response.choices[0].message.tool_calls:
                    tool_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    
                    # Execute the tool via MCP
                    tool_result = await self.execute_tool(tool_name, arguments)
                    
                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })
                
                # Get final response from OpenAI with tool results
                final_response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages
                )
                
                return final_response.choices[0].message.content or "No response generated"
            else:
                # No tools called - check if we should retry with retrieve_background
                if not forced_tool and any(word in question.lower() for word in ['eesha', 'experience', 'background', 'about']):
                    logging.info("No tools called, retrying with retrieve_background")
                    # Force retrieve_background tool
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        tools=self.tools,
                        tool_choice={"type": "function", "function": {"name": "retrieve_background"}},
                        temperature=0.7
                    )
                    
                    if response.choices[0].message.tool_calls:
                        # Process tool calls as before
                        assistant_message = {
                            "role": "assistant",
                            "content": response.choices[0].message.content or "",
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments
                                    }
                                }
                                for tc in response.choices[0].message.tool_calls
                            ]
                        }
                        messages.append(assistant_message)
                        
                        for tool_call in response.choices[0].message.tool_calls:
                            tool_name = tool_call.function.name
                            arguments = json.loads(tool_call.function.arguments)
                            tool_result = await self.execute_tool(tool_name, arguments)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_result
                            })
                        
                        final_response = self.openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=messages
                        )
                        
                        return final_response.choices[0].message.content or "No response generated"
                
                # No tools needed, return direct response
                return response.choices[0].message.content or "No response generated"
                
        except Exception as e:
            logging.error(f"Error in MCP client ask method: {str(e)}")
            return "I'm sorry, I encountered an error while processing your question. Please try again."

# Global persistent server instance
_mcp_server = None

def get_mcp_server():
    """Get or create the global MCP server instance"""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = PersistentMCPServer()
    return _mcp_server

# Synchronous wrapper for the async MCP client
class MCPAgent:
    def __init__(self):
        """Initialize the MCP agent with async client"""
        # Ensure server is running
        get_mcp_server()
        self.client = MCPOpenAIClient()
        
    def ask(self, question: str) -> str:
        """
        Synchronous interface for asking questions
        
        Args:
            question: The user's question
            
        Returns:
            AI response
        """
        try:
            # Run the async ask method
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.client.ask(question))
            finally:
                loop.close()
        except Exception as e:
            logging.error(f"Error in MCPAgent ask method: {str(e)}")
            return "I'm sorry, I encountered an error while processing your question. Please try again."

# Cleanup function to stop server on app shutdown
def cleanup():
    """Stop the MCP server on application shutdown"""
    global _mcp_server
    if _mcp_server:
        _mcp_server.stop_server()
        _mcp_server = None