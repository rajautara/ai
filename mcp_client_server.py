# requirements.txt
"""
fastmcp
mcp
asyncpg
pandas
PyPDF2
pyarrow
fastparquet
aiofiles
httpx
asyncio-mqtt
uvicorn
"""

# mcp_server.py - MCP Server Implementation
import asyncio
import json
import os
import pandas as pd
import PyPDF2
import asyncpg
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import aiofiles
from fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("Multi-Tool MCP Server")

# Database connection pool
db_pool = None

async def init_db_pool():
    """Initialize PostgreSQL connection pool"""
    global db_pool
    try:
        db_pool = await asyncpg.create_pool(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", 5432),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
            database=os.getenv("POSTGRES_DB", "postgres"),
            min_size=1,
            max_size=10
        )
        print("Database pool initialized successfully")
    except Exception as e:
        print(f"Failed to initialize database pool: {e}")
        db_pool = None

# PostgreSQL Tools
@mcp.tool()
async def query_postgresql(query: str, params: Optional[List] = None) -> Dict[str, Any]:
    """
    Execute a PostgreSQL query and return results.
    
    Args:
        query: SQL query to execute
        params: Optional parameters for the query
    
    Returns:
        Dictionary containing query results or error information
    """
    if not db_pool:
        return {"error": "Database connection not available"}
    
    try:
        async with db_pool.acquire() as conn:
            if params:
                result = await conn.fetch(query, *params)
            else:
                result = await conn.fetch(query)
            
            # Convert asyncpg Records to dictionaries
            rows = [dict(row) for row in result]
            
            return {
                "success": True,
                "data": rows,
                "row_count": len(rows)
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
async def get_table_schema(table_name: str, schema_name: str = "public") -> Dict[str, Any]:
    """
    Get the schema information for a PostgreSQL table.
    
    Args:
        table_name: Name of the table
        schema_name: Schema name (default: public)
    
    Returns:
        Dictionary containing table schema information
    """
    query = """
    SELECT column_name, data_type, is_nullable, column_default
    FROM information_schema.columns
    WHERE table_schema = $1 AND table_name = $2
    ORDER BY ordinal_position;
    """
    
    result = await query_postgresql(query, [schema_name, table_name])
    return result

@mcp.tool()
async def list_tables(schema_name: str = "public") -> Dict[str, Any]:
    """
    List all tables in a PostgreSQL schema.
    
    Args:
        schema_name: Schema name (default: public)
    
    Returns:
        Dictionary containing list of tables
    """
    query = """
    SELECT table_name, table_type
    FROM information_schema.tables
    WHERE table_schema = $1
    ORDER BY table_name;
    """
    
    result = await query_postgresql(query, [schema_name])
    return result

# Filesystem Tools
@mcp.tool()
async def read_csv_file(file_path: str, **kwargs) -> Dict[str, Any]:
    """
    Read a CSV file and return its contents.
    
    Args:
        file_path: Path to the CSV file
        **kwargs: Additional pandas read_csv parameters
    
    Returns:
        Dictionary containing CSV data and metadata
    """
    try:
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
        
        # Default parameters
        csv_params = {
            "encoding": "utf-8",
            "sep": ",",
            **kwargs
        }
        
        df = pd.read_csv(file_path, **csv_params)
        
        return {
            "success": True,
            "data": df.to_dict("records"),
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "dtypes": df.dtypes.to_dict(),
            "file_path": file_path
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
async def read_parquet_file(file_path: str) -> Dict[str, Any]:
    """
    Read a Parquet file and return its contents.
    
    Args:
        file_path: Path to the Parquet file
    
    Returns:
        Dictionary containing Parquet data and metadata
    """
    try:
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
        
        df = pd.read_parquet(file_path)
        
        return {
            "success": True,
            "data": df.to_dict("records"),
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "dtypes": df.dtypes.to_dict(),
            "file_path": file_path
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
async def read_pdf_file(file_path: str, page_range: Optional[tuple] = None) -> Dict[str, Any]:
    """
    Read a PDF file and extract text content.
    
    Args:
        file_path: Path to the PDF file
        page_range: Optional tuple (start, end) for page range
    
    Returns:
        Dictionary containing PDF text content
    """
    try:
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
        
        text_content = []
        
        async with aiofiles.open(file_path, 'rb') as file:
            pdf_content = await file.read()
            
        # Use PyPDF2 to extract text (running in thread pool for async)
        import io
        from concurrent.futures import ThreadPoolExecutor
        
        def extract_pdf_text(pdf_bytes):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            texts = []
            
            start_page = 0
            end_page = len(pdf_reader.pages)
            
            if page_range:
                start_page = max(0, page_range[0])
                end_page = min(len(pdf_reader.pages), page_range[1])
            
            for page_num in range(start_page, end_page):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                texts.append({
                    "page_number": page_num + 1,
                    "text": text
                })
            
            return texts
        
        with ThreadPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            text_content = await loop.run_in_executor(executor, extract_pdf_text, pdf_content)
        
        return {
            "success": True,
            "text_content": text_content,
            "total_pages": len(text_content),
            "file_path": file_path
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
async def list_directory(directory_path: str, file_extensions: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    List files in a directory, optionally filtered by file extensions.
    
    Args:
        directory_path: Path to the directory
        file_extensions: Optional list of file extensions to filter by
    
    Returns:
        Dictionary containing list of files
    """
    try:
        if not os.path.exists(directory_path):
            return {"error": f"Directory not found: {directory_path}"}
        
        if not os.path.isdir(directory_path):
            return {"error": f"Path is not a directory: {directory_path}"}
        
        files = []
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            
            if os.path.isfile(item_path):
                file_info = {
                    "name": item,
                    "path": item_path,
                    "size": os.path.getsize(item_path),
                    "extension": os.path.splitext(item)[1].lower()
                }
                
                if file_extensions:
                    if file_info["extension"] in [ext.lower() for ext in file_extensions]:
                        files.append(file_info)
                else:
                    files.append(file_info)
        
        return {
            "success": True,
            "files": files,
            "directory_path": directory_path,
            "file_count": len(files)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

# Server startup
async def startup_event():
    """Initialize database pool on startup"""
    await init_db_pool()
    print("MCP Server started successfully")

# Add startup event
mcp.add_event_handler("startup", startup_event)

if __name__ == "__main__":
    # Run the FastMCP server
    mcp.run(
        transport="sse",
        host="localhost", 
        port=8000
    )

# ================================
# mcp_client.py - MCP Client Implementation
# ================================

import asyncio
import json
import httpx
from typing import Dict, Any, List, Optional
from mcp import ClientSession
from mcp.client.sse import SseServerParameters

class CustomLLMAPI:
    """Custom LLM API client for the specified endpoint"""
    
    def __init__(self, api_key: str, apisid: str, apievn: str, apiempno: str):
        self.base_url = "https://telnout.sample.com/API/rest/chat/gpt4"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "apisid": apisid,
            "apievn": apievn,
            "apiempno": apiempno,
            "Content-Type": "application/json"
        }
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Send chat completion request to custom LLM API"""
        try:
            payload = {
                "messages": messages,
                **kwargs
            }
            
            response = await self.client.post(
                self.base_url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            return {
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

class MCPClient:
    """MCP Client that integrates with custom LLM API"""
    
    def __init__(self, llm_api: CustomLLMAPI, mcp_server_url: str = "http://localhost:8000"):
        self.llm_api = llm_api
        self.mcp_server_url = mcp_server_url
        self.mcp_session = None
        self.available_tools = {}
    
    async def connect_to_mcp_server(self):
        """Connect to the MCP server via SSE"""
        try:
            import httpx
            from mcp.client.sse import SseServerParameters
            
            # Create SSE server parameters
            server_params = SseServerParameters(
                url=f"{self.mcp_server_url}/sse"
            )
            
            # Create and start session
            self.mcp_session = ClientSession(server_params)
            await self.mcp_session.__aenter__()
            
            # Initialize the session
            init_result = await self.mcp_session.initialize()
            print(f"MCP Session initialized: {init_result}")
            
            # List available tools
            tools_response = await self.mcp_session.list_tools()
            self.available_tools = {tool.name: tool for tool in tools_response.tools}
            
            print(f"Connected to MCP server. Available tools: {list(self.available_tools.keys())}")
            return True
            
        except Exception as e:
            print(f"Failed to connect to MCP server: {e}")
            return False
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool on the MCP server"""
        try:
            if tool_name not in self.available_tools:
                return {"error": f"Tool '{tool_name}' not available"}
            
            result = await self.mcp_session.call_tool(tool_name, arguments)
            
            # Extract content from result
            if result.content:
                if hasattr(result.content[0], 'text'):
                    try:
                        # Try to parse as JSON if it's a string
                        content = result.content[0].text
                        return json.loads(content) if isinstance(content, str) else content
                    except json.JSONDecodeError:
                        return {"result": result.content[0].text}
                else:
                    return {"result": str(result.content[0])}
            else:
                return {"error": "No result returned"}
            
        except Exception as e:
            return {
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def process_user_query(self, user_message: str) -> str:
        """Process user query using LLM and MCP tools"""
        try:
            # System prompt that includes available tools
            system_prompt = f"""
            You are an AI assistant with access to the following MCP tools:
            {json.dumps(list(self.available_tools.keys()), indent=2)}
            
            Available tools:
            - PostgreSQL tools: query_postgresql, get_table_schema, list_tables
            - Filesystem tools: read_csv_file, read_parquet_file, read_pdf_file, list_directory
            
            When a user asks a question that requires data access, determine which tool(s) to use
            and provide the tool calls in a structured format. If you need to use a tool, 
            respond with a JSON object containing "tool_calls" array.
            
            Example response format for tool usage:
            {
                "tool_calls": [
                    {
                        "name": "query_postgresql",
                        "arguments": {"query": "SELECT * FROM users LIMIT 10"}
                    }
                ]
            }
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Get initial response from LLM
            llm_response = await self.llm_api.chat_completion(messages)
            
            if "error" in llm_response:
                return f"LLM API Error: {llm_response['error']}"
            
            # Extract the assistant's response
            assistant_message = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Check if the response contains tool calls
            try:
                parsed_response = json.loads(assistant_message)
                if "tool_calls" in parsed_response:
                    # Execute tools and get results
                    tool_results = []
                    for tool_call in parsed_response["tool_calls"]:
                        tool_name = tool_call["name"]
                        arguments = tool_call["arguments"]
                        
                        result = await self.execute_tool(tool_name, arguments)
                        tool_results.append({
                            "tool": tool_name,
                            "arguments": arguments,
                            "result": result
                        })
                    
                    # Send tool results back to LLM for final response
                    messages.append({"role": "assistant", "content": assistant_message})
                    messages.append({
                        "role": "user", 
                        "content": f"Tool execution results: {json.dumps(tool_results, indent=2)}\n\nPlease provide a natural language response based on these results."
                    })
                    
                    final_response = await self.llm_api.chat_completion(messages)
                    if "error" not in final_response:
                        return final_response.get("choices", [{}])[0].get("message", {}).get("content", "No response generated")
                    else:
                        return f"Error generating final response: {final_response['error']}"
                
            except json.JSONDecodeError:
                # Response is not JSON, return as-is
                pass
            
            return assistant_message
            
        except Exception as e:
            return f"Error processing query: {e}"
    
    async def interactive_session(self):
        """Run an interactive session with the user"""
        print("MCP Client Interactive Session")
        print("Type 'quit' to exit, 'tools' to list available tools")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\nUser: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'tools':
                    print(f"Available tools: {list(self.available_tools.keys())}")
                    continue
                elif not user_input:
                    continue
                
                print("Assistant: Processing...")
                response = await self.process_user_query(user_input)
                print(f"Assistant: {response}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    
    async def close(self):
        """Close connections"""
        if self.mcp_session:
            try:
                await self.mcp_session.__aexit__(None, None, None)
            except Exception as e:
                print(f"Error closing MCP session: {e}")
        await self.llm_api.close()

# Example usage
async def main():
    # Initialize custom LLM API client
    llm_api = CustomLLMAPI(
        api_key="your-api-key",
        apisid="your-apisid",
        apievn="your-apievn",
        apiempno="your-apiempno"
    )
    
    # Initialize MCP client
    mcp_client = MCPClient(llm_api)
    
    # Connect to MCP server
    if await mcp_client.connect_to_mcp_server():
        # Run interactive session
        await mcp_client.interactive_session()
    else:
        print("Failed to connect to MCP server")
    
    # Clean up
    await mcp_client.close()

if __name__ == "__main__":
    asyncio.run(main())

# ================================
# config.py - Configuration file
# ================================

import os
from typing import Dict, Any

# Database configuration
DATABASE_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
    "database": os.getenv("POSTGRES_DB", "postgres"),
}

# MCP Server configuration
MCP_SERVER_CONFIG = {
    "host": os.getenv("MCP_HOST", "localhost"),
    "port": int(os.getenv("MCP_PORT", 8000)),
}

# Custom LLM API configuration
LLM_API_CONFIG = {
    "api_key": os.getenv("LLM_API_KEY", ""),
    "apisid": os.getenv("LLM_APISID", ""
