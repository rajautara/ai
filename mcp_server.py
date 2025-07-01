#!/usr/bin/env python3
"""
MCP Server with PostgreSQL and File System capabilities
"""

import asyncio
import asyncpg
import pandas as pd
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)
import PyPDF2
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPServer:
    def __init__(self):
        self.server = Server("postgresql-filesystem-server")
        self.db_pool: Optional[asyncpg.Pool] = None
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup MCP server handlers"""
        
        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """List available resources"""
            return [
                Resource(
                    uri="postgresql://tables",
                    name="PostgreSQL Tables",
                    description="List all tables in PostgreSQL database",
                    mimeType="application/json"
                ),
                Resource(
                    uri="filesystem://",
                    name="File System",
                    description="Access to file system for CSV, PDF, Parquet files",
                    mimeType="application/json"
                )
            ]
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read a specific resource"""
            if uri == "postgresql://tables":
                return await self.list_database_tables()
            elif uri.startswith("filesystem://"):
                path = uri.replace("filesystem://", "")
                return await self.read_file_info(path)
            else:
                raise ValueError(f"Unknown resource: {uri}")
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="execute_sql",
                    description="Execute SQL query on PostgreSQL database",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "SQL query to execute"
                            },
                            "params": {
                                "type": "array",
                                "description": "Query parameters",
                                "items": {"type": "string"},
                                "default": []
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="read_csv",
                    description="Read and analyze CSV file",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to CSV file"
                            },
                            "nrows": {
                                "type": "integer",
                                "description": "Number of rows to read (optional)",
                                "default": None
                            }
                        },
                        "required": ["file_path"]
                    }
                ),
                Tool(
                    name="read_parquet",
                    description="Read and analyze Parquet file",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to Parquet file"
                            }
                        },
                        "required": ["file_path"]
                    }
                ),
                Tool(
                    name="read_pdf",
                    description="Extract text from PDF file",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to PDF file"
                            },
                            "page_limit": {
                                "type": "integer",
                                "description": "Maximum number of pages to read",
                                "default": 5
                            }
                        },
                        "required": ["file_path"]
                    }
                ),
                Tool(
                    name="list_files",
                    description="List files in a directory",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "directory": {
                                "type": "string",
                                "description": "Directory path to list files from"
                            },
                            "file_types": {
                                "type": "array",
                                "description": "File extensions to filter (e.g., ['.csv', '.pdf'])",
                                "items": {"type": "string"},
                                "default": [".csv", ".pdf", ".parquet"]
                            }
                        },
                        "required": ["directory"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls"""
            try:
                if name == "execute_sql":
                    result = await self.execute_sql(
                        arguments["query"], 
                        arguments.get("params", [])
                    )
                elif name == "read_csv":
                    result = await self.read_csv_file(
                        arguments["file_path"],
                        arguments.get("nrows")
                    )
                elif name == "read_parquet":
                    result = await self.read_parquet_file(arguments["file_path"])
                elif name == "read_pdf":
                    result = await self.read_pdf_file(
                        arguments["file_path"],
                        arguments.get("page_limit", 5)
                    )
                elif name == "list_files":
                    result = await self.list_directory_files(
                        arguments["directory"],
                        arguments.get("file_types", [".csv", ".pdf", ".parquet"])
                    )
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                return [TextContent(type="text", text=str(result))]
            
            except Exception as e:
                logger.error(f"Error executing tool {name}: {str(e)}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def initialize_database(self, database_url: str):
        """Initialize database connection pool"""
        try:
            self.db_pool = await asyncpg.create_pool(database_url)
            logger.info("Database connection pool created")
        except Exception as e:
            logger.error(f"Failed to create database pool: {str(e)}")
            raise
    
    async def execute_sql(self, query: str, params: List[str] = None) -> Dict[str, Any]:
        """Execute SQL query"""
        if not self.db_pool:
            raise RuntimeError("Database not initialized")
        
        params = params or []
        async with self.db_pool.acquire() as conn:
            try:
                if query.strip().upper().startswith('SELECT'):
                    rows = await conn.fetch(query, *params)
                    return {
                        "type": "query_result",
                        "rows": [dict(row) for row in rows],
                        "row_count": len(rows)
                    }
                else:
                    result = await conn.execute(query, *params)
                    return {
                        "type": "execution_result",
                        "result": result,
                        "message": "Query executed successfully"
                    }
            except Exception as e:
                return {
                    "type": "error",
                    "error": str(e)
                }
    
    async def list_database_tables(self) -> str:
        """List all tables in the database"""
        if not self.db_pool:
            return json.dumps({"error": "Database not initialized"})
        
        query = """
        SELECT table_name, table_type 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
        """
        
        result = await self.execute_sql(query)
        return json.dumps(result, indent=2)
    
    async def read_csv_file(self, file_path: str, nrows: Optional[int] = None) -> Dict[str, Any]:
        """Read and analyze CSV file"""
        try:
            df = pd.read_csv(file_path, nrows=nrows)
            return {
                "file_path": file_path,
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "head": df.head().to_dict(orient="records"),
                "summary": df.describe().to_dict() if len(df.select_dtypes(include='number').columns) > 0 else "No numeric columns"
            }
        except Exception as e:
            return {"error": f"Failed to read CSV: {str(e)}"}
    
    async def read_parquet_file(self, file_path: str) -> Dict[str, Any]:
        """Read and analyze Parquet file"""
        try:
            df = pd.read_parquet(file_path)
            return {
                "file_path": file_path,
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "head": df.head().to_dict(orient="records"),
                "summary": df.describe().to_dict() if len(df.select_dtypes(include='number').columns) > 0 else "No numeric columns"
            }
        except Exception as e:
            return {"error": f"Failed to read Parquet: {str(e)}"}
    
    async def read_pdf_file(self, file_path: str, page_limit: int = 5) -> Dict[str, Any]:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                text_content = []
                pages_to_read = min(page_limit, num_pages)
                
                for page_num in range(pages_to_read):
                    page = pdf_reader.pages[page_num]
                    text_content.append({
                        "page": page_num + 1,
                        "text": page.extract_text()
                    })
                
                return {
                    "file_path": file_path,
                    "total_pages": num_pages,
                    "pages_read": pages_to_read,
                    "content": text_content
                }
        except Exception as e:
            return {"error": f"Failed to read PDF: {str(e)}"}
    
    async def list_directory_files(self, directory: str, file_types: List[str]) -> Dict[str, Any]:
        """List files in directory with specified extensions"""
        try:
            path = Path(directory)
            if not path.exists():
                return {"error": f"Directory does not exist: {directory}"}
            
            files = []
            for file_path in path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in [ext.lower() for ext in file_types]:
                    files.append({
                        "name": file_path.name,
                        "path": str(file_path),
                        "size": file_path.stat().st_size,
                        "extension": file_path.suffix
                    })
            
            return {
                "directory": directory,
                "file_types": file_types,
                "files": files,
                "count": len(files)
            }
        except Exception as e:
            return {"error": f"Failed to list files: {str(e)}"}
    
    async def read_file_info(self, path: str) -> str:
        """Get file information"""
        try:
            file_path = Path(path)
            if file_path.exists():
                info = {
                    "path": str(file_path),
                    "exists": True,
                    "is_file": file_path.is_file(),
                    "size": file_path.stat().st_size if file_path.is_file() else None,
                    "extension": file_path.suffix if file_path.is_file() else None
                }
            else:
                info = {"path": path, "exists": False}
            
            return json.dumps(info, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

async def main():
    """Main server function"""
    # Initialize server
    mcp_server = MCPServer()
    
    # Initialize database (replace with your actual database URL)
    database_url = "postgresql://username:password@localhost:5432/database_name"
    try:
        await mcp_server.initialize_database(database_url)
    except Exception as e:
        logger.warning(f"Database initialization failed: {e}")
        logger.info("Server will run without database functionality")
    
    # Run server
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.server.run(
            read_stream,
            write_stream,
            mcp_server.server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())