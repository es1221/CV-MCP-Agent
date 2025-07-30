# CV Agent - AI-Powered Resume Assistant

## Overview

This is a Flask-based web application that serves as an AI-powered CV agent for Eesha Sondhi. The application uses OpenAI's GPT-4 through Model Context Protocol (MCP) architecture to answer questions about professional background, experience, and projects using natural language processing and vector similarity search with a modern web interface.

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes

### July 23, 2025
- **Project Cleanup and Enhancement**: Cleaned up codebase for deployment readiness
  - Fixed JSON syntax errors in `repo_summaries.json` (lines 36-44)
  - Removed unnecessary files from `attached_assets` folder and old cache files
  - Enhanced `summarise_repo()` function to include GitHub repository links
  - Updated system prompts to emphasize including GitHub links for specific project queries
  - Improved project descriptions with proper formatting and feature lists
  - Replaced redundant `list_all_repos` tool with comprehensive `get_all_projects` tool
  - Updated cv-agent features in repo_summaries.json to reflect current MCP-based architecture
  - Project now deployment-ready with clean file structure

### July 22, 2025
- **Code Cleanup and Simplification**: Removed unused components for cleaner codebase
  - Deleted TTS functionality (`tts.py`) and all audio-related features
  - Removed legacy OpenAI Assistant implementation (`agent.py`) 
  - Cleaned up frontend to remove audio player components
  - Removed `attached_assets` folder containing outdated code snippets
  - Simplified Flask app responses to text-only format
  - Updated documentation to reflect current architecture

### July 21, 2025
- **Implemented Model Context Protocol (MCP) Architecture**: Complete transformation from direct OpenAI Assistant API to MCP-based architecture
  - Created MCP server (`mcp_server.py`) exposing CV tools as standardized MCP tools
  - Built MCP client (`mcp_client.py`) integrating OpenAI with MCP server through stdio transport
  - Updated Flask app to use MCP-based agent instead of direct Assistant API
  - MCP provides standardized interface between AI models and backend systems
  - Supports all existing tools: retrieve_background, summarise_repo, get_education, get_skills, get_personal_info, get_butterfly_air_info
- **Performance Optimization Implementation**: Resolved critical embedding regeneration issue
  - **Persistent FAISS Index**: Embeddings cached to disk with change detection (133 documents)
  - **Persistent MCP Server**: Single server process maintains state across tool calls
  - **Pre-generation Script**: `pregenerate_embeddings.py` creates cached embeddings upfront
  - **Proper Cleanup**: Clean server shutdown prevents zombie processes
  - **Result**: System loads in seconds instead of minutes, embeddings only regenerate when documents change

### July 18, 2025
- **Added Dedicated Butterfly Air Tool**: Created `get_butterfly_air_info()` function to provide comprehensive information about Eesha's work at Butterfly Air
- **Improved Consistency**: Dedicated tool ensures reliable retrieval of work experience details including role, achievements, technical implementations, and business impact
- **Enhanced Agent Configuration**: Updated system prompt to prioritize Butterfly Air tool for all company-related queries

## System Architecture

### Backend Architecture
- **Framework**: Flask (Python) web framework for HTTP handling
- **AI Integration**: Model Context Protocol (MCP) architecture with OpenAI GPT-4
  - MCP Server: Exposes CV tools as standardized MCP tools
  - MCP Client: Handles OpenAI <-> MCP server communication via stdio transport
  - Function Calling: Dynamic tool execution through MCP protocol
- **Vector Search**: FAISS (Facebook AI Similarity Search) for document embedding and retrieval
- **Document Processing**: Custom embedding system using OpenAI's text-embedding-3-small model

### Frontend Architecture
- **UI Framework**: Bootstrap 5 with dark theme
- **JavaScript**: Vanilla JavaScript for chat interface
- **Responsive Design**: Mobile-first approach with Bootstrap grid system

## Key Components

### 1. MCP Server (mcp_server.py)
- Exposes CV tools as standardized MCP tools using FastMCP framework
- Tools: retrieve_background, summarise_repo, get_education, get_skills, get_personal_info, get_butterfly_air_info, list_all_repos
- Handles tool execution and error management
- Communicates via stdio transport protocol

### 2. MCP Client (mcp_client.py)
- MCPOpenAIClient: Manages OpenAI <-> MCP server communication
- Converts MCP tools to OpenAI function format dynamically
- Handles async tool execution and response processing
- MCPAgent: Synchronous wrapper for Flask integration



### 2. DocumentEmbedder (embed.py)
- Handles document chunking and embedding generation
- Uses FAISS for efficient vector similarity search
- Processes CV, dissertation, and project documents from `/data` directory

### 3. CVTools (tools.py)
- Provides tool functions for the AI assistant
- `retrieve_background()`: Searches embedded documents for relevant information
- `summarise_repo()`: Returns GitHub repository summaries from JSON data
- `get_butterfly_air_info()`: Dedicated tool for comprehensive Butterfly Air work experience queries

### 4. Flask Application (app.py)
- RESTful API endpoint `/ask` for processing questions
- Returns text responses from MCP-based CV agent
- Error handling and logging

## Data Flow

1. **User Input**: User submits question through web interface
2. **Question Processing**: Flask app receives question via `/ask` endpoint
3. **AI Processing**: MCP-based agent processes question using OpenAI GPT-4
4. **Tool Execution**: Agent calls appropriate MCP tools (retrieve_background, summarise_repo)
5. **Document Search**: Tools use FAISS to find relevant information from embedded documents
6. **Response Generation**: Assistant generates natural language response
7. **Response Delivery**: Text response returned to frontend
8. **UI Update**: Frontend displays response with formatted text

## External Dependencies

### APIs
- **OpenAI API**: GPT-4O for conversation management and embeddings generation

### Libraries
- **Flask**: Web framework
- **FAISS**: Vector similarity search
- **OpenAI Python SDK**: API client
- **NumPy**: Numerical operations for embeddings
- **Bootstrap 5**: Frontend styling
- **Font Awesome**: Icons
- **MarkItDown**: Enhanced PDF and document processing for better text extraction

### Environment Variables
- `OPENAI_API_KEY`: Required for OpenAI API access
- `SESSION_SECRET`: Flask session security

## Deployment Strategy

### Local Development
- Single-process Flask application with debug mode
- File-based document storage in `/data` directory
- In-memory FAISS index initialization

### Production Considerations
- Environment variable management for API keys
- Static file serving through Flask
- Error handling and logging for production debugging
- Potential for containerization with Docker

### Data Management
- **Caching System**: PDF processing results cached as markdown files (`cv.md`, `masters_dissertation.md`)
- **Optimization**: Cached files eliminate redundant PDF processing on application restart
- **Document Embeddings**: Generated at startup using cached markdown content
- **Repository Summaries**: Stored in JSON format for quick access
- **Static Document Files**: CV, dissertation, and project files in `/data` directory
- **MarkItDown Integration**: Enhanced PDF document processing and text extraction
- **Authentic Data**: Real CV and dissertation PDFs processed for accurate information retrieval
- **Semantic Chunking**: LangChain RecursiveCharacterTextSplitter with 1000-character chunks and 200-character overlap
- **Similarity Threshold**: Optimized threshold (0.3) for improved document retrieval accuracy

## Security Considerations

### API Key Management
- Environment variables for sensitive credentials
- No hardcoded API keys in source code

### Input Validation
- Question length and content validation
- Error handling for malformed requests

### Data Privacy
- Local document processing without external data sharing
- Conversation data not persisted beyond request lifecycle