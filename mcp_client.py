#!/usr/bin/env python3
"""
FastMCP Client that connects to FastMCP server and custom LLM API
"""

import asyncio
import httpx
import json
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomLLMClient:
    """Client for custom LLM API"""
    
    def __init__(self, api_url: str, api_key: Optional[str] = None):
        self.api_url = api_url
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Send chat completion request to custom LLM API"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "messages": messages,
            **kwargs
        }
        
        try:
            response = await self.client.post(
                self.api_url,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            raise
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

class FastMCPClient:
    """Client for FastMCP server"""
    
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)
        self.available_tools: List[Dict[str, Any]] = []
        self.available_resources: List[Dict[str, Any]] = []
    
    async def connect(self) -> bool:
        """Connect to FastMCP server and get capabilities"""
        try:
            # Get server info
            response = await self.client.get(f"{self.server_url}/mcp/info")
            response.raise_for_status()
            server_info = response.json()
            logger.info(f"Connected to server: {server_info.get('name', 'Unknown')}")
            
            # Get available tools
            tools_response = await self.client.get(f"{self.server_url}/mcp/tools")
            tools_response.raise_for_status()
            self.available_tools = tools_response.json()
            
            # Get available resources
            resources_response = await self.client.get(f"{self.server_url}/mcp/resources")
            resources_response.raise_for_status()
            self.available_resources = resources_response.json()
            
            logger.info(f"Found {len(self.available_tools)} tools and {len(self.available_resources)} resources")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to FastMCP server: {str(e)}")
            return False
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the FastMCP server"""
        try:
            payload = {
                "name": tool_name,
                "arguments": arguments
            }
            
            response = await self.client.post(
                f"{self.server_url}/mcp/tools/call",
                json=payload
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to call tool {tool_name}: {str(e)}")
            return {"error": f"Tool call failed: {str(e)}"}
    
    async def get_resource(self, resource_uri: str) -> Dict[str, Any]:
        """Get a resource from the FastMCP server"""
        try:
            response = await self.client.get(
                f"{self.server_url}/mcp/resources/read",
                params={"uri": resource_uri}
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get resource {resource_uri}: {str(e)}")
            return {"error": f"Resource fetch failed: {str(e)}"}
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

class MCPClient:
    """MCP Client that orchestrates LLM and FastMCP server interactions"""
    
    def __init__(self, llm_api_url: str, mcp_server_url: str = "http://localhost:8000", llm_api_key: Optional[str] = None):
        self.llm_client = CustomLLMClient(llm_api_url, llm_api_key)
        self.mcp_client = FastMCPClient(mcp_server_url)
        self.conversation_history: List[Dict[str, str]] = []
    
    async def connect_to_mcp_server(self) -> bool:
        """Connect to FastMCP server"""
        return await self.mcp_client.connect()
    
    async def get_tools_description(self) -> str:
        """Get description of available MCP tools"""
        if not self.mcp_client.available_tools:
            return "No tools available"
        
        tools_desc = "Available tools:\n"
        for tool in self.mcp_client.available_tools:
            tools_desc += f"- {tool['name']}: {tool['description']}\n"
            if 'inputSchema' in tool and tool['inputSchema']:
                # Simplify schema description
                props = tool['inputSchema'].get('properties', {})
                if props:
                    params = [f"{k}: {v.get('description', 'no description')}" for k, v in props.items()]
                    tools_desc += f"  Parameters: {', '.join(params)}\n"
        
        return tools_desc
    
    async def get_resources_description(self) -> str:
        """Get description of available resources"""
        if not self.mcp_client.available_resources:
            return "No resources available"
        
        resources_desc = "Available resources:\n"
        for resource in self.mcp_client.available_resources:
            resources_desc += f"- {resource['uri']}: {resource['name']} - {resource['description']}\n"
        
        return resources_desc
    
    def parse_tool_calls_from_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse tool calls from LLM response using JSON format"""
        tool_calls = []
        
        # Look for JSON tool calls in the response
        lines = response.split('\n')
        current_call = None
        
        for line in lines:
            line = line.strip()
            
            # Look for tool call markers
            if line.startswith('```json') or line.startswith('```tool_call'):
                current_call = ""
            elif line == '```' and current_call is not None:
                # End of tool call block
                try:
                    tool_data = json.loads(current_call)
                    if 'tool' in tool_data and 'arguments' in tool_data:
                        tool_calls.append({
                            "name": tool_data['tool'],
                            "arguments": tool_data['arguments']
                        })
                except json.JSONDecodeError:
                    pass
                current_call = None
            elif current_call is not None:
                current_call += line + '\n'
            
            # Also look for simple format
            elif line.startswith('TOOL:'):
                tool_name = line.replace('TOOL:', '').strip()
                # Look for arguments in following lines
                args = {}
                tool_calls.append({"name": tool_name, "arguments": args})
        
        # Alternative: look for structured patterns
        if not tool_calls:
            # Simple pattern matching for tool calls
            import re
            
            # Pattern: use_tool(tool_name, {arguments})
            pattern = r'use_tool\s*\(\s*["\']([^"\']+)["\'],\s*({[^}]*})\s*\)'
            matches = re.findall(pattern, response)
            
            for tool_name, args_str in matches:
                try:
                    args = json.loads(args_str)
                    tool_calls.append({"name": tool_name, "arguments": args})
                except json.JSONDecodeError:
                    tool_calls.append({"name": tool_name, "arguments": {}})
        
        return tool_calls
    
    async def execute_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute MCP tool and return formatted result"""
        result = await self.mcp_client.call_tool(tool_name, arguments)
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        # Format the result nicely
        if isinstance(result, dict):
            # Remove metadata for cleaner output
            clean_result = {k: v for k, v in result.items() if not k.startswith('_')}
            return json.dumps(clean_result, indent=2)
        
        return str(result)
    
    async def process_user_query(self, user_query: str) -> str:
        """Process user query with LLM and MCP tools"""
        # Build context
        tools_desc = await self.get_tools_description()
        resources_desc = await self.get_resources_description()
        
        system_message = f"""You are an AI assistant with access to database and file system tools.

{tools_desc}

{resources_desc}

When you need to use a tool, you can call it by including a JSON block in your response:

```json
{{
  "tool": "tool_name",
  "arguments": {{
    "param1": "value1",
    "param2": "value2"
  }}
}}
```

You can make multiple tool calls. Always explain what you're doing and interpret results for the user.

Available tools:
- execute_sql: Run SQL queries on PostgreSQL database
- read_csv: Analyze CSV files
- read_parquet: Read Parquet files  
- read_pdf: Extract text from PDFs
- list_files: List files in directories
- get_table_schema: Get database table structure
- analyze_file_content: Analyze any supported file type

Always provide helpful, clear responses and explain the results of any tool calls.
"""
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_message},
            *self.conversation_history[-10:],  # Keep last 10 messages for context
            {"role": "user", "content": user_query}
        ]
        
        try:
            # Get initial LLM response
            llm_response = await self.llm_client.chat_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )
            
            assistant_message = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Check for tool calls
            tool_calls = self.parse_tool_calls_from_response(assistant_message)
            
            if tool_calls:
                # Execute tools
                tool_results = []
                for tool_call in tool_calls:
                    logger.info(f"Executing tool: {tool_call['name']} with args: {tool_call['arguments']}")
                    result = await self.execute_mcp_tool(tool_call["name"], tool_call["arguments"])
                    tool_results.append({
                        "tool": tool_call["name"],
                        "arguments": tool_call["arguments"],
                        "result": result
                    })
                
                # Send results back to LLM
                tool_summary = "\n\n".join([
                    f"Tool: {tr['tool']}\nArguments: {json.dumps(tr['arguments'])}\nResult: {tr['result']}"
                    for tr in tool_results
                ])
                
                final_messages = messages + [
                    {"role": "assistant", "content": assistant_message},
                    {"role": "user", "content": f"Here are the tool execution results:\n\n{tool_summary}\n\nPlease provide a final response interpreting these results."}
                ]
                
                final_response = await self.llm_client.chat_completion(
                    messages=final_messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                final_content = final_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # Update conversation history
                self.conversation_history.extend([
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": final_content}
                ])
                
                return final_content
            else:
                # No tools needed
                self.conversation_history.extend([
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": assistant_message}
                ])
                
                return assistant_message
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Error processing your request: {str(e)}"
    
    async def close(self):
        """Close all connections"""
        await self.llm_client.close()
        await self.mcp_client.close()

async def interactive_chat():
    """Interactive chat interface"""
    print("Initializing MCP Client...")
    
    # Initialize client
    client = MCPClient(
        llm_api_url="https://telnout.sample.com/API/rest/chat/gpt4",
        mcp_server_url="http://localhost:8000",
        llm_api_key="your-api-key-here"  # Replace with actual API key
    )
    
    # Connect to FastMCP server
    print("Connecting to FastMCP server...")
    if not await client.connect_to_mcp_server():
        print("Failed to connect to FastMCP server. Make sure it's running on http://localhost:8000")
        return
    
    print("✓ Connected to FastMCP server!")
    print("\nAvailable commands:")
    print("- Ask questions about your data")
    print("- Type 'tools' to see available tools")
    print("- Type 'resources' to see available resources")
    print("- Type 'quit' to exit")
    print("=" * 60)
    
    try:
        while True:
            user_input = input("\n🤖 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
            elif user_input.lower() == 'tools':
                tools_desc = await client.get_tools_description()
                print(f"\n📋 Available Tools:\n{tools_desc}")
            elif user_input.lower() == 'resources':
                resources_desc = await client.get_resources_description()
                print(f"\n📂 Available Resources:\n{resources_desc}")
            elif user_input:
                print("\n🤔 Processing...")
                response = await client.process_user_query(user_input)
                print(f"\n🤖 Assistant: {response}")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    finally:
        await client.close()

async def example_queries():
    """Run example queries"""
    client = MCPClient(
        llm_api_url="https://telnout.sample.com/API/rest/chat/gpt4",
        mcp_server_url="http://localhost:8000",
        llm_api_key="your-api-key-here"
    )
    
    if not await client.connect_to_mcp_server():
        print("Failed to connect to FastMCP server")
        return
    
    queries = [
        "What files are available in the current directory?",
        "List all tables in the database",
        "Show me the structure of the products table",
        "Read and analyze any CSV file you can find",
        "Count the total number of rows in all database tables"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Example {i} ---")
        print(f"Query: {query}")
        response = await client.process_user_query(query)
        print(f"Response: {response}")
        print("-" * 50)
    
    await client.close()

if __name__ == "__main__":
    print("FastMCP Client")
    print("1. Interactive chat")
    print("2. Run example queries")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "2":
        asyncio.run(example_queries())
    else:
        asyncio.run(interactive_chat())
