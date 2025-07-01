#!/usr/bin/env python3
"""
Corrected FastMCP Server - Understanding the actual FastMCP API
"""

import asyncio
import asyncpg
import pandas as pd
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import PyPDF2

# Import configuration
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global database pool
db_pool: Optional[asyncpg.Pool] = None

async def initialize_database():
    """Initialize database connection pool"""
    global db_pool
    try:
        db_pool = await asyncpg.create_pool(config.database_url)
        logger.info("Database connection pool created")
    except Exception as e:
        logger.error(f"Failed to create database pool: {str(e)}")
        logger.info("Server will run without database functionality")

async def cleanup_database():
    """Cleanup database connections"""
    global db_pool
    if db_pool:
        await db_pool.close()
        logger.info("Database pool closed")

# Try to determine the correct FastMCP usage
try:
    from fastmcp import FastMCP
    
    # Initialize FastMCP
    app = FastMCP()
    
    # Define tools using FastMCP decorators
    @app.tool()
    async def execute_sql(query: str, params: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute SQL query on PostgreSQL database
        
        Args:
            query: SQL query to execute
            params: Query parameters (optional)
        """
        if not db_pool:
            return {"error": "Database not initialized"}
        
        params = params or []
        
        try:
            async with db_pool.acquire() as conn:
                if query.strip().upper().startswith('SELECT'):
                    rows = await conn.fetch(query, *params)
                    return {
                        "type": "query_result",
                        "rows": [dict(row) for row in rows],
                        "row_count": len(rows),
                        "query": query
                    }
                else:
                    result = await conn.execute(query, *params)
                    return {
                        "type": "execution_result",
                        "result": result,
                        "message": "Query executed successfully",
                        "query": query
                    }
        except Exception as e:
            logger.error(f"SQL execution error: {str(e)}")
            return {
                "type": "error",
                "error": str(e),
                "query": query
            }
    
    @app.tool()
    async def read_csv(file_path: str, nrows: Optional[int] = None) -> Dict[str, Any]:
        """
        Read and analyze CSV file
        
        Args:
            file_path: Path to CSV file
            nrows: Number of rows to read (optional)
        """
        try:
            if not Path(file_path).exists():
                return {"error": f"File not found: {file_path}"}
            
            df = pd.read_csv(file_path, nrows=nrows)
            
            info = {
                "file_path": file_path,
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "head": df.head().to_dict(orient="records")
            }
            
            # Add summary statistics for numeric columns
            numeric_cols = df.select_dtypes(include='number').columns
            if len(numeric_cols) > 0:
                info["summary_statistics"] = df[numeric_cols].describe().to_dict()
            
            # Add missing value information
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                info["missing_values"] = missing_values[missing_values > 0].to_dict()
            
            return info
            
        except Exception as e:
            logger.error(f"CSV reading error: {str(e)}")
            return {"error": f"Failed to read CSV: {str(e)}"}
    
    @app.tool()
    async def read_parquet(file_path: str) -> Dict[str, Any]:
        """
        Read and analyze Parquet file
        
        Args:
            file_path: Path to Parquet file
        """
        try:
            if not Path(file_path).exists():
                return {"error": f"File not found: {file_path}"}
            
            df = pd.read_parquet(file_path)
            
            info = {
                "file_path": file_path,
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "head": df.head().to_dict(orient="records")
            }
            
            # Add summary statistics for numeric columns
            numeric_cols = df.select_dtypes(include='number').columns
            if len(numeric_cols) > 0:
                info["summary_statistics"] = df[numeric_cols].describe().to_dict()
            
            return info
            
        except Exception as e:
            logger.error(f"Parquet reading error: {str(e)}")
            return {"error": f"Failed to read Parquet: {str(e)}"}
    
    @app.tool()
    async def read_pdf(file_path: str, page_limit: int = 5) -> Dict[str, Any]:
        """
        Extract text from PDF file
        
        Args:
            file_path: Path to PDF file
            page_limit: Maximum number of pages to read
        """
        try:
            if not Path(file_path).exists():
                return {"error": f"File not found: {file_path}"}
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                content = []
                pages_to_read = min(page_limit, num_pages)
                
                for page_num in range(pages_to_read):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    content.append({
                        "page": page_num + 1,
                        "text": text,
                        "char_count": len(text)
                    })
                
                return {
                    "file_path": file_path,
                    "total_pages": num_pages,
                    "pages_read": pages_to_read,
                    "content": content,
                    "total_characters": sum(page["char_count"] for page in content)
                }
                
        except Exception as e:
            logger.error(f"PDF reading error: {str(e)}")
            return {"error": f"Failed to read PDF: {str(e)}"}
    
    @app.tool()
    async def list_files(directory: str = ".", file_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        List files in a directory
        
        Args:
            directory: Directory path to list files from
            file_types: File extensions to filter (e.g., ['.csv', '.pdf'])
        """
        if file_types is None:
            file_types = [".csv", ".pdf", ".parquet", ".json", ".xlsx"]
        
        try:
            path = Path(directory)
            if not path.exists():
                return {"error": f"Directory does not exist: {directory}"}
            
            if not path.is_dir():
                return {"error": f"Path is not a directory: {directory}"}
            
            files = []
            for file_path in path.iterdir():
                if file_path.is_file():
                    file_ext = file_path.suffix.lower()
                    if file_ext in [ext.lower() for ext in file_types]:
                        stat = file_path.stat()
                        files.append({
                            "name": file_path.name,
                            "path": str(file_path.absolute()),
                            "size": stat.st_size,
                            "extension": file_ext,
                            "modified": stat.st_mtime
                        })
            
            # Sort by name
            files.sort(key=lambda x: x["name"])
            
            return {
                "directory": str(path.absolute()),
                "file_types_filter": file_types,
                "files": files,
                "count": len(files),
                "total_size": sum(f["size"] for f in files)
            }
            
        except Exception as e:
            logger.error(f"Directory listing error: {str(e)}")
            return {"error": f"Failed to list files: {str(e)}"}
    
    @app.tool()
    async def get_table_schema(table_name: str) -> Dict[str, Any]:
        """
        Get schema information for a specific table
        
        Args:
            table_name: Name of the table to describe
        """
        if not db_pool:
            return {"error": "Database not initialized"}
        
        try:
            async with db_pool.acquire() as conn:
                # Get column information
                column_query = """
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length,
                    numeric_precision,
                    numeric_scale
                FROM information_schema.columns 
                WHERE table_name = $1 AND table_schema = 'public'
                ORDER BY ordinal_position
                """
                
                columns = await conn.fetch(column_query, table_name)
                
                if not columns:
                    return {"error": f"Table '{table_name}' not found"}
                
                # Get row count (approximate for large tables)
                count_query = f"SELECT COUNT(*) FROM {table_name}"
                row_count = await conn.fetchval(count_query)
                
                return {
                    "table_name": table_name,
                    "columns": [dict(col) for col in columns],
                    "row_count": row_count
                }
                
        except Exception as e:
            logger.error(f"Schema query error: {str(e)}")
            return {"error": f"Failed to get table schema: {str(e)}"}
    
    fastmcp_available = True
    logger.info("FastMCP imported successfully")

except ImportError as e:
    logger.error(f"FastMCP import failed: {e}")
    fastmcp_available = False
    app = None

# Alternative implementation using standard web framework
if not fastmcp_available:
    logger.info("Using alternative web server implementation")
    
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    
    app = FastAPI(title="MCP Server", description="PostgreSQL and File System MCP Server")
    
    class ToolCall(BaseModel):
        name: str
        arguments: dict
    
    class ToolResponse(BaseModel):
        result: dict
    
    # Tool implementations (same as above but as regular functions)
    async def execute_sql_impl(query: str, params: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute SQL query implementation"""
        if not db_pool:
            return {"error": "Database not initialized"}
        
        params = params or []
        
        try:
            async with db_pool.acquire() as conn:
                if query.strip().upper().startswith('SELECT'):
                    rows = await conn.fetch(query, *params)
                    return {
                        "type": "query_result",
                        "rows": [dict(row) for row in rows],
                        "row_count": len(rows),
                        "query": query
                    }
                else:
                    result = await conn.execute(query, *params)
                    return {
                        "type": "execution_result",
                        "result": result,
                        "message": "Query executed successfully",
                        "query": query
                    }
        except Exception as e:
            logger.error(f"SQL execution error: {str(e)}")
            return {
                "type": "error",
                "error": str(e),
                "query": query
            }
    
    async def read_csv_impl(file_path: str, nrows: Optional[int] = None) -> Dict[str, Any]:
        """Read CSV implementation"""
        try:
            if not Path(file_path).exists():
                return {"error": f"File not found: {file_path}"}
            
            df = pd.read_csv(file_path, nrows=nrows)
            
            info = {
                "file_path": file_path,
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "head": df.head().to_dict(orient="records")
            }
            
            numeric_cols = df.select_dtypes(include='number').columns
            if len(numeric_cols) > 0:
                info["summary_statistics"] = df[numeric_cols].describe().to_dict()
            
            return info
            
        except Exception as e:
            logger.error(f"CSV reading error: {str(e)}")
            return {"error": f"Failed to read CSV: {str(e)}"}
    
    async def list_files_impl(directory: str = ".", file_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """List files implementation"""
        if file_types is None:
            file_types = [".csv", ".pdf", ".parquet", ".json", ".xlsx"]
        
        try:
            path = Path(directory)
            if not path.exists():
                return {"error": f"Directory does not exist: {directory}"}
            
            files = []
            for file_path in path.iterdir():
                if file_path.is_file():
                    file_ext = file_path.suffix.lower()
                    if file_ext in [ext.lower() for ext in file_types]:
                        stat = file_path.stat()
                        files.append({
                            "name": file_path.name,
                            "path": str(file_path.absolute()),
                            "size": stat.st_size,
                            "extension": file_ext
                        })
            
            return {
                "directory": str(path.absolute()),
                "files": files,
                "count": len(files)
            }
            
        except Exception as e:
            logger.error(f"Directory listing error: {str(e)}")
            return {"error": f"Failed to list files: {str(e)}"}
    
    # FastAPI endpoints
    @app.get("/")
    async def root():
        return {"message": "MCP Server", "status": "running"}
    
    @app.get("/health")
    async def health():
        return {"status": "healthy"}
    
    @app.get("/tools")
    async def list_tools():
        return [
            {
                "name": "execute_sql",
                "description": "Execute SQL query on PostgreSQL database",
                "parameters": {
                    "query": "string",
                    "params": "array (optional)"
                }
            },
            {
                "name": "read_csv",
                "description": "Read and analyze CSV file",
                "parameters": {
                    "file_path": "string",
                    "nrows": "integer (optional)"
                }
            },
            {
                "name": "list_files",
                "description": "List files in a directory",
                "parameters": {
                    "directory": "string (default: '.')",
                    "file_types": "array (optional)"
                }
            }
        ]
    
    @app.post("/tools/call")
    async def call_tool(tool_call: ToolCall):
        tool_name = tool_call.name
        arguments = tool_call.arguments
        
        try:
            if tool_name == "execute_sql":
                result = await execute_sql_impl(
                    arguments.get("query", ""),
                    arguments.get("params", [])
                )
            elif tool_name == "read_csv":
                result = await read_csv_impl(
                    arguments.get("file_path", ""),
                    arguments.get("nrows")
                )
            elif tool_name == "list_files":
                result = await list_files_impl(
                    arguments.get("directory", "."),
                    arguments.get("file_types")
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

async def run_server():
    """Run the server"""
    await initialize_database()
    logger.info("Server initialized")
    
    if fastmcp_available:
        logger.info("Starting FastMCP server")
        try:
            # Try different FastMCP run methods
            if hasattr(app, 'run'):
                await app.run(host=config.SERVER_HOST, port=config.SERVER_PORT)
            elif hasattr(app, 'serve'):
                await app.serve(host=config.SERVER_HOST, port=config.SERVER_PORT)
            else:
                # Run as stdio-based MCP server
                logger.info("Running FastMCP as stdio server")
                await app.run()
        except Exception as e:
            logger.error(f"FastMCP server failed: {e}")
    else:
        logger.info(f"Starting FastAPI server on {config.SERVER_HOST}:{config.SERVER_PORT}")
        import uvicorn
        config_obj = uvicorn.Config(
            app,
            host=config.SERVER_HOST,
            port=config.SERVER_PORT,
            log_level="info"
        )
        server = uvicorn.Server(config_obj)
        await server.serve()

if __name__ == "__main__":
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        asyncio.run(cleanup_database())
