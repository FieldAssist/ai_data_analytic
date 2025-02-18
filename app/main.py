import re
import os
import json
import logging
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime
from dateutil import parser
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import openai
import clickhouse_connect
from clickhouse_connect import get_client
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure OpenAI
logger.info("Configuring OpenAI...")
openai.api_type = "azure"
openai.api_version = "2024-02-15-preview"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-35-turbo")

app = FastAPI(title="AI Data Analyst API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error handler caught: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__,
            "path": str(request.url.path)
        }
    )

class Query(BaseModel):
    question: str

def get_clickhouse_client():
    """Get a configured ClickHouse client."""
    try:
        host = os.getenv("CLICKHOUSE_HOST")
        port = int(os.getenv("CLICKHOUSE_PORT", "8123"))
        user = os.getenv("CLICKHOUSE_USER")
        password = os.getenv("CLICKHOUSE_PASSWORD")
        database = os.getenv("CLICKHOUSE_DATABASE")
        
        logger.info(f"ClickHouse Configuration:")
        logger.info(f"- Host: {host}")
        logger.info(f"- Port: {port}")
        logger.info(f"- User: {user}")
        logger.info(f"- Database: {database}")
        
        if not all([host, user, password, database]):
            logger.error(f"Missing ClickHouse configuration. Got: host={host}, user={user}, database={database}")
            raise ValueError("Missing required ClickHouse configuration. Please check your .env file.")
            
        logger.info(f"Attempting to connect to ClickHouse at {host}:{port}")
        try:
            client = get_client(
                host=host,
                port=port,
                username=user,
                password=password,
                database=database,
                connect_timeout=10
            )
            # Test the connection with database info
            result = client.query("SELECT currentDatabase(), version()")
            db, version = result.result_rows[0]
            logger.info(f"Successfully connected to ClickHouse {version} using database '{db}'")
            return client
        except Exception as conn_err:
            logger.error(f"Failed to connect to ClickHouse: {str(conn_err)}")
            logger.error(f"Connection error details: {traceback.format_exc()}")
            raise
    except Exception as e:
        logger.error(f"Error in get_clickhouse_client: {str(e)}")
        logger.error(f"Full error details: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Database connection error: {str(e)}\nPlease check your database configuration and ensure the server is accessible."
        )

class ClickHouseQueryGenerator:
    def __init__(self):
        logger.info("Initializing ClickHouseQueryGenerator...")
        self.client = get_clickhouse_client()
        self.schema_cache = {}
        self.table_stats_cache = {}
        
    def execute_query(self, query: str, description: str = "") -> Any:
        """Execute a query with logging and handle any data type."""
        try:
            logger.info(f"Executing query: {description}")
            logger.debug(f"Original SQL: {query}")
            
            # Clean the query
            query = query.strip()
            
            # Remove any FORMAT clause
            format_index = query.upper().rfind('FORMAT')
            if format_index != -1:
                query = query[:format_index].strip()
            
            # Remove any trailing semicolons
            query = query.rstrip(';')
            
            # Split on semicolon and take only the first statement
            query = query.split(';')[0].strip()
            
            logger.debug(f"Cleaned SQL: {query}")
            
            start_time = datetime.now()
            
            try:
                # First try with JSON format to handle all types
                json_query = f"{query} FORMAT JSONEachRow"
                result = self.client.query(json_query)
                
                # Parse the JSON result
                result_data = {
                    "columns": result.column_names,
                    "rows": []
                }
                
                for row in result.result_rows:
                    formatted_row = []
                    for val in row:
                        formatted_row.append(val)
                    result_data["rows"].append(formatted_row)
                    
            except Exception as format_error:
                logger.warning(f"JSON format failed, trying native format: {format_error}")
                
                # Fallback to native format with safe conversion
                result = self.client.query(query)
                
                result_data = {
                    "columns": result.column_names,
                    "rows": []
                }
                
                for row in result.result_rows:
                    formatted_row = []
                    for val in row:
                        # Safe conversion of any type
                        try:
                            if val is None:
                                formatted_row.append(None)
                            elif isinstance(val, (int, float, bool)):
                                formatted_row.append(val)
                            elif isinstance(val, (bytes, bytearray)):
                                formatted_row.append(val.hex())
                            else:
                                formatted_row.append(str(val))
                        except Exception as e:
                            logger.warning(f"Error converting value {val}: {e}")
                            formatted_row.append(str(val))
                    result_data["rows"].append(formatted_row)
            
            duration = (datetime.now() - start_time).total_seconds()
            row_count = len(result_data["rows"])
            logger.info(f"Query completed in {duration:.2f}s, returned {row_count} rows")
            
            return result_data
            
        except Exception as e:
            logger.error(f"Query failed: {description}")
            logger.error(f"SQL: {query}")
            logger.error(f"Error: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return error in a structured format
            return {
                "error": str(e),
                "query": query,
                "columns": [],
                "rows": []
            }

    def get_table_schema(self, table_name: str) -> Dict[str, Dict[str, str]]:
        """Get detailed schema information for a table."""
        if table_name in self.schema_cache:
            return self.schema_cache[table_name]
        
        try:
            logger.info(f"Getting schema for table: {table_name}")
            query = f"""
            SELECT 
                name,
                type,
                default_expression,
                comment,
                is_in_primary_key,
                is_in_partition_key,
                compression_codec
            FROM system.columns 
            WHERE table = '{table_name}'
            AND database = currentDatabase()
            """
            # Remove newlines and extra spaces
            query = ' '.join(query.split())
            
            result = self.execute_query(query, f"Get schema for {table_name}")
            rows = result["rows"]
            
            if not rows:
                logger.warning(f"No schema found for table {table_name}")
                return None
            
            schema = {}
            for row in rows:
                name, type_, default, comment, is_pk, is_part, codec = row
                schema[name] = {
                    'type': type_,
                    'default': default,
                    'comment': comment,
                    'is_primary_key': bool(is_pk),
                    'is_partition_key': bool(is_part),
                    'compression': codec
                }
            
            logger.info(f"Found {len(schema)} columns for table {table_name}")
            self.schema_cache[table_name] = schema
            return schema
            
        except Exception as e:
            logger.error(f"Error getting schema for table {table_name}: {e}")
            return None

    def get_table_statistics(self, table_name: str) -> Dict[str, Any]:
        """Get detailed statistics about a table using MCP ClickHouse."""
        if table_name in self.table_stats_cache:
            return self.table_stats_cache[table_name]
            
        try:
            # Get table statistics
            table_query = f"""
            SELECT 
                name as table,
                total_rows,
                total_bytes,
                engine,
                partition_key,
                sorting_key,
                primary_key,
                sampling_key
            FROM system.tables 
            WHERE name = '{table_name}'
            AND database = currentDatabase()
            """
            table_result = self.execute_query(table_query, f"Get stats for {table_name}")
            
            if not table_result["rows"]:
                return None
                
            row = table_result["rows"][0]
            stats = {
                'name': row[0],
                'total_rows': row[1],
                'total_bytes': row[2],
                'engine': row[3],
                'partition_key': row[4],
                'sorting_key': row[5],
                'primary_key': row[6],
                'sampling_key': row[7],
                'column_stats': {}
            }
            
            # Get column information
            schema = self.get_table_schema(table_name)
            if not schema:
                return None
                
            # Get numeric columns
            numeric_columns = []
            for col, info in schema.items():
                if any(t in info['type'].lower() for t in ['int', 'float', 'decimal']):
                    numeric_columns.append(col)
            
            # Get statistics for numeric columns
            if numeric_columns:
                stats_query = f"""
                SELECT
                    {','.join(f'''
                        min({col}) as min_{col},
                        max({col}) as max_{col},
                        avg({col}) as avg_{col},
                        count() as count_{col},
                        uniqExact({col}) as unique_{col}
                    ''' for col in numeric_columns)}
                FROM {table_name}
                """
                try:
                    stats_result = self.execute_query(stats_query, f"Get column stats for {table_name}")
                    if stats_result["rows"]:
                        row = stats_result["rows"][0]
                        for i, col in enumerate(numeric_columns):
                            base_idx = i * 5  # 5 stats per column
                            stats['column_stats'][col] = {
                                'min': row[base_idx] if row[base_idx] is not None else 0,
                                'max': row[base_idx + 1] if row[base_idx + 1] is not None else 0,
                                'avg': row[base_idx + 2] if row[base_idx + 2] is not None else 0,
                                'count': row[base_idx + 3] if row[base_idx + 3] is not None else 0,
                                'unique_count': row[base_idx + 4] if row[base_idx + 4] is not None else 0
                            }
                except Exception as e:
                    logger.error(f"Error getting column stats for {table_name}: {e}")
                    pass
            
            self.table_stats_cache[table_name] = stats
            return stats
            
        except Exception as e:
            logger.error(f"Error getting table statistics for {table_name}: {e}")
            return None

    def clean_sql_query(self, query: str) -> str:
        """Clean and validate SQL query."""
        # Remove any trailing semicolons
        query = query.strip().rstrip(';')
        
        # Remove any FORMAT clause that might have been added
        format_index = query.upper().find('FORMAT')
        if format_index != -1:
            query = query[:format_index].strip()
            
        # Ensure it starts with SELECT
        if not query.upper().startswith('SELECT'):
            raise ValueError("Query must start with SELECT")
            
        # Add LIMIT if not present
        if 'LIMIT' not in query.upper():
            query += ' LIMIT 1000'
            
        return query

    def generate_query(self, question: str) -> str:
        """Generate a SQL query based on the question."""
        try:
            logger.info(f"Generating query for question: {question}")
            
            # Get available tables info for context
            tables_query = """
            SELECT name, engine, total_rows
            FROM system.tables 
            WHERE database = currentDatabase()
            """
            tables_result = self.execute_query(tables_query, "Get available tables")
            
            # Build context about available tables
            table_context = []
            for row in tables_result["rows"]:
                table, engine, rows = row
                # Get schema for the table
                schema = self.get_table_schema(table)
                if schema:
                    columns = list(schema.keys())
                    table_context.append(f"Table '{table}' ({rows} rows) with columns: {', '.join(columns)}")
            
            table_info = "\n".join(table_context)
            
            # Prepare the prompt
            prompt = f"""You are a ClickHouse SQL expert. Generate a SQL query to answer this question: {question}

Available tables and their schemas:
{table_info}

Rules:
1. Use only the tables and columns listed above
2. For better performance, only select needed columns
3. Add FINAL if using ReplacingMergeTree tables
4. Use proper aggregations when needed
5. Keep the query simple and efficient
6. DO NOT add semicolons or FORMAT clauses
7. DO NOT use multiple statements
8. If dealing with timestamps, use toDateTime() for conversion
9. For date/time operations, use proper ClickHouse functions

Return only the SQL query, nothing else."""

            logger.info("Sending request to OpenAI...")
            response = openai.ChatCompletion.create(
                engine=OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are a ClickHouse SQL expert. Generate only SQL queries, no explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=500
            )
            
            sql_query = response.choices[0].message.content.strip()
            logger.info(f"Raw generated SQL query: {sql_query}")
            
            # Clean and validate the query
            sql_query = self.clean_sql_query(sql_query)
            logger.info(f"Cleaned SQL query: {sql_query}")
            
            return sql_query
            
        except Exception as e:
            logger.error(f"Error generating query: {e}")
            logger.error(traceback.format_exc())
            raise

    def optimize_query(self, query: str) -> str:
        """Optimize the generated query for ClickHouse."""
        try:
            # Add FINAL modifier for tables with ReplacingMergeTree engine
            tables_query = "SELECT name, engine FROM system.tables WHERE database = currentDatabase()"
            tables_result = self.execute_query(tables_query, "Get table engines")
            
            for row in tables_result["rows"]:
                table, engine = row
                if 'ReplacingMergeTree' in engine and f'FROM {table}' in query and 'FINAL' not in query:
                    query = query.replace(f'FROM {table}', f'FROM {table} FINAL')
            
            return query
        except Exception as e:
            logger.error(f"Error optimizing query: {e}")
            return query

    def explain_query(self, query: str) -> str:
        """Get query explanation from ClickHouse."""
        try:
            explain_query = f"EXPLAIN pipeline {query}"
            result = self.execute_query(explain_query, "Explain query")
            return "\n".join(row[0] for row in result["rows"])
        except Exception as e:
            logger.error(f"Error explaining query: {e}")
            return ""

@app.post("/analyze")
async def analyze_data(query: Query):
    """Analyze data based on the question."""
    try:
        logger.info(f"Analyzing data for question: {query.question}")
        
        # Initialize query generator
        logger.info("Initializing query generator...")
        query_generator = ClickHouseQueryGenerator()
        
        # Generate and execute the query
        sql_query = query_generator.generate_query(query.question)
        result = query_generator.execute_query(sql_query, "Main analysis query")
        
        # Check for errors
        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=f"Query failed: {result['error']}"
            )
        
        # Get query explanation
        explanation_prompt = f"""Explain this SQL query in simple terms:
{sql_query}

Explain:
1. What data it's retrieving
2. Any calculations or aggregations
3. Any filters or conditions
4. The expected output

Keep it concise and user-friendly."""

        explanation_response = openai.ChatCompletion.create(
            engine=OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a SQL expert explaining queries to users."},
                {"role": "user", "content": explanation_prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        explanation = explanation_response.choices[0].message.content.strip()
        
        # Format response
        response = {
            "query": sql_query,
            "explanation": explanation,
            "columns": result["columns"],
            "rows": result["rows"],
            "message": f"Found {len(result['rows'])} results"
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error in analyze_data: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/tables")
async def get_tables():
    """Get detailed information about available tables."""
    try:
        logger.info("Fetching table information...")
        query_generator = ClickHouseQueryGenerator()
        tables = []
        tables_query = "SELECT DISTINCT name as table FROM system.tables WHERE database = currentDatabase()"
        tables_result = query_generator.execute_query(tables_query, "Get available tables")
        
        for row in tables_result["rows"]:
            table_name = row[0]
            try:
                stats = query_generator.get_table_statistics(table_name)
                if stats:
                    tables.append(stats)
            except Exception as e:
                logger.error(f"Error getting stats for table {table_name}: {e}")
                continue
                
        logger.info(f"Found {len(tables)} tables")
        return {"tables": tables}
    except Exception as e:
        logger.error(f"Error getting tables: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get tables: {str(e)}"
        )

@app.post("/analyze_query")
async def analyze_query(query: str):
    """Analyze a SQL query without executing it."""
    try:
        logger.info(f"Analyzing query: {query}")
        query_generator = ClickHouseQueryGenerator()
        explanation = query_generator.explain_query(query)
        logger.info(f"Query explanation: {explanation}")
        return {
            "query": query,
            "explanation": explanation
        }
    except Exception as e:
        logger.error(f"Error analyzing query: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze query: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint to verify server and database connection."""
    try:
        # Check database connection
        client = get_clickhouse_client()
        result = client.query("SELECT 1")
        if result.result_rows:
            logger.info("Health check: Database connection successful")
            return {
                "status": "healthy",
                "database": "connected",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify API is working."""
    logger.info("Test endpoint called")
    return {"status": "ok", "message": "API is working"}

@app.get("/test-db")
async def test_db():
    """Test database connection and return basic info."""
    logger.info("Testing database connection...")
    try:
        client = get_clickhouse_client()
        result = client.query("SELECT currentDatabase(), version()")
        db, version = result.result_rows[0]
        
        # Get list of tables
        tables_result = client.query("SELECT name FROM system.tables WHERE database = currentDatabase()")
        tables = [row[0] for row in tables_result.result_rows]
        
        return {
            "status": "ok",
            "database": db,
            "version": version,
            "tables": tables
        }
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/test-openai")
async def test_openai():
    """Test OpenAI connection."""
    logger.info("Testing OpenAI connection...")
    try:
        # Test OpenAI configuration
        logger.info("OpenAI Configuration:")
        logger.info(f"- API Type: {openai.api_type}")
        logger.info(f"- API Version: {openai.api_version}")
        logger.info(f"- API Base: {openai.api_base}")
        logger.info(f"- Deployment: {OPENAI_DEPLOYMENT}")
        
        # Test a simple completion
        logger.info("Making test API call...")
        response = openai.ChatCompletion.create(
            engine=OPENAI_DEPLOYMENT,
            messages=[
                {"role": "user", "content": "Say hello"}
            ],
            max_tokens=10
        )
        
        return {
            "status": "ok",
            "message": response.choices[0].message.content,
            "model": response.model,
            "usage": response.usage._asdict()
        }
    except Exception as e:
        logger.error(f"OpenAI test failed: {e}")
        logger.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
