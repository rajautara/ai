#!/usr/bin/env python3
"""
MCP Client that connects to MCP server and custom LLM API
"""

import asyncio
import httpx
import json
from typing import Dict, List, Any, Optional
import logging
from mcp.client import ClientSession
from mcp.client.stdio import stdio_client
from mcp.types import TextContent, Tool
import subprocess
import sys

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

class MCPClient:
    """MCP Client that orchestrates LLM and MCP server interactions"""
    
    def __init__(self, llm_api_url: str, llm_api_key: Optional[str] = None):
        self.llm_client = CustomLLMClient(llm_api_url, llm_api_key)
        self.mcp_session: Optional[ClientSession] = None
        self.available_tools: List[Tool] = []
        self.conversation_history: List[Dict[str, str]] = []
    
    async def connect_to_mcp_server(self, server_script_path: str):
        """Connect to MCP server"""
        try:
            # Start MCP server process
            server_process = await asyncio.create_subprocess_exec(
                sys.executable, server_script_path,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Create client session
            read_stream, write_stream = stdio_client(server_process)
            self.mcp_session = ClientSession(read_stream, write_stream)
            
            # Initialize session
            await self.mcp_session.initialize()
            
            # Get available tools
            self.available_tools = await self.mcp_session.list_tools()
            logger.info(f"Connected to MCP server with {len(self.available_tools)} tools")
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {str(e)}")
            return False
    
    async def get_tools_description(self) -> str:
        """Get description of available MCP tools"""
        if not self.available_tools:
            return "No tools available"
        
        tools_desc = "Available tools:\n"
        for tool in self.available_tools:
            tools_desc += f"- {tool.name}: {tool.description}\n"
            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                tools_desc += f"  Parameters: {json.dumps(tool.inputSchema, indent=2)}\n"
        
        return tools_desc
    
    async def execute_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute MCP tool and return result"""
        if not self.mcp_session:
            return "Error: Not connected to MCP server"
        
        try:
            result = await self.mcp_session.call_tool(tool_name, arguments)
            if result and len(result) > 0:
                return result[0].text if hasattr(result[0], 'text') else str(result[0])
            return "No result returned"
        except Exception as e:
            logger.error(f"Failed to execute tool {tool_name}: {str(e)}")
            return f"Error executing tool: {str(e)}"
    
    def extract_tool_calls(self, llm_response: str) -> List[Dict[str, Any]]:
        """Extract tool calls from LLM response"""
        tool_calls = []
        
        # Simple parsing - you might want to implement more sophisticated parsing
        # depending on your LLM's response format
        lines = llm_response.split('\n')
        current_tool = None
        current_args = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('TOOL_CALL:'):
                if current_tool:
                    tool_calls.append({"name": current_tool, "arguments": current_args})
                current_tool = line.replace('TOOL_CALL:', '').strip()
                current_args = {}
            elif line.startswith('ARG:') and current_tool:
                parts = line.replace('ARG:', '').strip().split('=', 1)
                if len(parts) == 2:
                    key, value = parts
                    try:
                        # Try to parse as JSON
                        current_args[key.strip()] = json.loads(value.strip())
                    except:
                        # Fall back to string
                        current_args[key.strip()] = value.strip()
        
        if current_tool:
            tool_calls.append({"name": current_tool, "arguments": current_args})
        
        return tool_calls
    
    async def process_user_query(self, user_query: str) -> str:
        """Process user query with LLM and MCP tools"""
        # Add tools context to the system message
        tools_desc = await self.get_tools_description()
        system_message = f"""You are an AI assistant with access to the following tools:

{tools_desc}

When you need to use a tool, respond with:
TOOL_CALL: tool_name
ARG: parameter_name=parameter_value
ARG: another_parameter="value"

You can make multiple tool calls in sequence. Always explain what you're doing and interpret the results for the user.
"""
        
        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": system_message},
            *self.conversation_history,
            {"role": "user", "content": user_query}
        ]
        
        try:
            # Get initial LLM response
            llm_response = await self.llm_client.chat_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            # Extract response content
            assistant_message = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Check for tool calls
            tool_calls = self.extract_tool_calls(assistant_message)
            
            if tool_calls:
                # Execute tools and get results
                tool_results = []
                for tool_call in tool_calls:
                    result = await self.execute_mcp_tool(
                        tool_call["name"], 
                        tool_call["arguments"]
                    )
                    tool_results.append({
                        "tool": tool_call["name"],
                        "result": result
                    })
                
                # Send tool results back to LLM for final response
                tool_results_text = "\n".join([
                    f"Tool {tr['tool']} result: {tr['result']}" 
                    for tr in tool_results
                ])
                
                final_messages = messages + [
                    {"role": "assistant", "content": assistant_message},
                    {"role": "user", "content": f"Tool execution results:\n{tool_results_text}\n\nPlease provide a final response based on these results."}
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
                # No tool calls needed
                self.conversation_history.extend([
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": assistant_message}
                ])
                
                return assistant_message
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Error: {str(e)}"
    
    async def close(self):
        """Close connections"""
        if self.llm_client:
            await self.llm_client.close()
        if self.mcp_session:
            await self.mcp_session.close()

async def interactive_chat():
    """Interactive chat interface"""
    # Initialize client
    client = MCPClient(
        llm_api_url="https://telnout.sample.com/API/rest/chat/gpt4",
        llm_api_key="your-api-key-here"  # Replace with actual API key if needed
    )
    
    # Connect to MCP server
    server_connected = await client.connect_to_mcp_server("mcp_server.py")
    if not server_connected:
        print("Failed to connect to MCP server. Exiting.")
        return
    
    print("MCP Client connected! Available commands:")
    print("- Type your questions")
    print("- Type 'quit' to exit")
    print("- Type 'tools' to see available tools")
    print("-" * 50)
    
    try:
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
            elif user_input.lower() == 'tools':
                tools_desc = await client.get_tools_description()
                print(f"\nAssistant: {tools_desc}")
            elif user_input:
                print("\nThinking...")
                response = await client.process_user_query(user_input)
                print(f"\nAssistant: {response}")
    
    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        await client.close()

async def example_usage():
    """Example of programmatic usage"""
    client = MCPClient(
        llm_api_url="https://telnout.sample.com/API/rest/chat/gpt4",
        llm_api_key="your-api-key-here"
    )
    
    # Connect to MCP server
    if await client.connect_to_mcp_server("mcp_server.py"):
        # Example queries
        queries = [
            "List all files in the current directory",
            "What tables are available in the database?",
            "Read the first 10 rows of data.csv if it exists"
        ]
        
        for query in queries:
            print(f"\nQuery: {query}")
            response = await client.process_user_query(query)
            print(f"Response: {response}")
    
    await client.close()

if __name__ == "__main__":
    # You can choose to run either interactive chat or example usage
    print("Starting MCP Client...")
    print("1. Interactive chat")
    print("2. Example usage")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(interactive_chat())
    else:
        asyncio.run(example_usage())