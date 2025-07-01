#!/usr/bin/env python3
"""
FastMCP Server with PostgreSQL and File System capabilities
"""

import asyncio
import asyncpg
import pandas as pd
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
from fastmcp import FastMCP
import PyPDF2
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP
mcp = FastMCP("PostgreSQL-FileSystem-Server")

# Global database pool
db_pool: Optional[asyncpg.Pool] = None

# Configuration
DATABASE_URL = "postgresql://username:password@localhost:5432/database_name"
SERVER_HOST = "localhost"
SERVER_PORT = 8000

async def initialize_database():
    """Initialize database connection pool"""
    global db_pool
    try:
        db_pool = await asyncpg.create_pool(DATABASE_URL)
        logger.info("Database connection pool created")
    except Exception as e:
        logger.error(f"Failed to create database pool: {str(e)}")
        logger.info("Server will run without database functionality")

@mcp.resource("postgresql://tables")
async def list_database_tables() -> str:
    """List all tables in PostgreSQL database"""
    if not db_pool:
        return json.dumps({"error": "Database not initialized"})
    
    query = """
    SELECT table_name, table_type, table_schema
    FROM information_schema.tables 
    WHERE table_schema = 'public'
    ORDER BY table_name
    """
    
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(query)
            tables = [dict(row) for row in rows]
            return json.dumps({
                "type": "database_tables",
                "tables": tables,
                "count": len(tables)
            }, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to list tables: {str(e)}"})

@mcp.resource("filesystem://info")
async def filesystem_info() -> str:
    """Get filesystem information"""
    current_dir = Path.cwd()
    supported_types = [".csv", ".pdf", ".parquet", ".json", ".xlsx"]
    
    return json.dumps({
        "current_directory": str(current_dir),
        "supported_file_types": supported_types,
        "description": "File system access for data files"
    }, indent=2)

@mcp.tool()
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

@mcp.tool()
async def read_csv(file_path: str, nrows: Optional[int] = None) -> Dict[str, Any]:
    """
    Read and analyze CSV file
    
    Args:
        file_path: Path to CSV file
        nrows: Number of rows to read (optional)
    """
    try:
        # Check if file exists
        if not Path(file_path).exists():
            return {"error": f"File not found: {file_path}"}
        
        df = pd.read_csv(file_path, nrows=nrows)
        
        # Get basic info
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

@mcp.tool()
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

@mcp.tool()
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

@mcp.tool()
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

@mcp.tool()
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
            
            # Get table size information
            size_query = """
            SELECT 
                pg_size_pretty(pg_total_relation_size($1)) as total_size,
                pg_size_pretty(pg_relation_size($1)) as table_size,
                (SELECT count(*) FROM information_schema.tables WHERE table_name = $1) as exists
            """
            
            size_info = await conn.fetchrow(size_query, table_name)
            
            # Get row count (approximate for large tables)
            count_query = f"SELECT COUNT(*) FROM {table_name}"
            row_count = await conn.fetchval(count_query)
            
            return {
                "table_name": table_name,
                "columns": [dict(col) for col in columns],
                "row_count": row_count,
                "total_size": size_info["total_size"] if size_info else "Unknown",
                "table_size": size_info["table_size"] if size_info else "Unknown"
            }
            
    except Exception as e:
        logger.error(f"Schema query error: {str(e)}")
        return {"error": f"Failed to get table schema: {str(e)}"}

@mcp.tool()
async def analyze_file_content(file_path: str) -> Dict[str, Any]:
    """
    Analyze file content and provide insights
    
    Args:
        file_path: Path to file to analyze
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}
        
        file_ext = path.suffix.lower()
        
        if file_ext == ".csv":
            return await read_csv(file_path, nrows=1000)  # Sample for large files
        elif file_ext == ".parquet":
            return await read_parquet(file_path)
        elif file_ext == ".pdf":
            return await read_pdf(file_path, page_limit=3)
        elif file_ext == ".json":
            with open(file_path, 'r') as f:
                data = json.load(f)
                return {
                    "file_path": file_path,
                    "type": "json",
                    "keys": list(data.keys()) if isinstance(data, dict) else "Not a JSON object",
                    "size": len(str(data)),
                    "sample": str(data)[:500] + "..." if len(str(data)) > 500 else str(data)
                }
        else:
            return {"error": f"Unsupported file type: {file_ext}"}
            
    except Exception as e:
        logger.error(f"File analysis error: {str(e)}")
        return {"error": f"Failed to analyze file: {str(e)}"}

async def startup():
    """Startup tasks"""
    await initialize_database()
    logger.info("FastMCP Server initialized")

async def shutdown():
    """Cleanup tasks"""
    global db_pool
    if db_pool:
        await db_pool.close()
        logger.info("Database pool closed")

if __name__ == "__main__":
    # Set up the FastMCP app
    app = mcp.create_app()
    
    # Add startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        await startup()
    
    @app.on_event("shutdown")
    async def shutdown_event():
        await shutdown()
    
    # Run the server
    logger.info(f"Starting FastMCP server on {SERVER_HOST}:{SERVER_PORT}")
    uvicorn.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info"
        )
