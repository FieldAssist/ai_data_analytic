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
from decimal import Decimal
from dateutil import parser

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
        
    def get_relevant_knowledge(self, query: str) -> Dict[str, Any]:
        """Get relevant knowledge from knowledge base"""
        try:
            # Get recent knowledge entries
            result = self.client.query("""
                SELECT timestamp, knowledge_id, raw_knowledge, analysis, patterns
                FROM knowledge_base
                ORDER BY timestamp DESC
                LIMIT 5
            """)
            
            if not result.result_rows:
                return {}
            
            # Combine knowledge entries
            knowledge_base = {
                row[1]: {
                    'timestamp': row[0],
                    'raw_knowledge': json.loads(row[2]),
                    'analysis': row[3],
                    'patterns': json.loads(row[4])
                }
                for row in result.result_rows
            }
            
            # Use Azure OpenAI to find relevant knowledge
            messages = [
                {"role": "system", "content": "Find and return the most relevant knowledge patterns for the given query. Focus on query patterns, metrics, and business rules that could help answer the question."},
                {"role": "user", "content": f"Query: {query}\nKnowledge Base: {json.dumps(knowledge_base)}"}
            ]
            
            response = openai.ChatCompletion.create(
                engine=OPENAI_DEPLOYMENT,
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error retrieving knowledge: {str(e)}")
            return {}

    def learn_from_interaction(self, question: str, generated_query: str, results: Any, performance_metrics: Dict[str, Any]) -> None:
        """Learn from each interaction to improve future responses"""
        try:
            # Create new knowledge entry
            interaction_knowledge = {
                "query_pattern": {
                    "question_type": question,
                    "generated_sql": generated_query,
                    "performance": performance_metrics
                },
                "results_analysis": {
                    "data_patterns": results if isinstance(results, (dict, list)) else str(results)[:1000],
                    "metrics_used": performance_metrics.get("metrics_used", [])
                }
            }
            
            timestamp = datetime.now()
            knowledge_id = f"k_learned_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Analyze the interaction
            messages = [
                {"role": "system", "content": "Analyze this interaction and extract patterns, insights, and improvements for future queries."},
                {"role": "user", "content": f"Interaction: {json.dumps(interaction_knowledge)}"}
            ]
            
            analysis = openai.ChatCompletion.create(
                engine=OPENAI_DEPLOYMENT,
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            ).choices[0].message.content
            
            # Extract patterns
            patterns = openai.ChatCompletion.create(
                engine=OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "Convert the analysis into a structured JSON format with patterns, rules, and metrics."},
                    {"role": "user", "content": analysis}
                ],
                temperature=0.1,
                max_tokens=1000
            ).choices[0].message.content
            
            # Store in knowledge base
            self.client.command("""
                INSERT INTO knowledge_base (
                    timestamp, knowledge_id, raw_knowledge, analysis, patterns
                ) VALUES (
                    %(timestamp)s, %(knowledge_id)s, %(raw_knowledge)s, %(analysis)s, %(patterns)s
                )
            """, parameters={
                'timestamp': timestamp,
                'knowledge_id': knowledge_id,
                'raw_knowledge': json.dumps(interaction_knowledge),
                'analysis': analysis,
                'patterns': patterns
            })
            
            logger.info(f"Learned from interaction. Knowledge ID: {knowledge_id}")
            
        except Exception as e:
            logger.error(f"Error learning from interaction: {str(e)}")

    def execute_query(self, query: str, description: str = "") -> Any:
        """Execute a query with logging and handle any data type."""
        try:
            logger.info(f"Executing query: {description}\n{query}")
            
            # Execute the query
            result = self.client.query(query)
            
            # Convert result to list of dictionaries
            columns = result.column_names
            rows = []
            
            for row in result.result_rows:
                row_dict = {}
                for i, col in enumerate(columns):
                    # Handle different data types
                    value = row[i]
                    if hasattr(value, 'isoformat'):  # Handle any datetime-like object
                        value = value.isoformat()
                    elif isinstance(value, Decimal):
                        value = float(value)
                    row_dict[col] = value
                rows.append(row_dict)
            
            logger.info(f"Query executed successfully. Found {len(rows)} rows")
            return rows
            
        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            logger.error(f"{error_msg}\nQuery: {query}")
            raise Exception(error_msg)

    def get_table_schema(self, table_name: str) -> Dict[str, str]:
        """Get detailed schema information for a table."""
        try:
            if table_name in self.schema_cache:
                return self.schema_cache[table_name]
            
            # Query table schema
            schema_query = f"""
            DESCRIBE {table_name}
            """
            
            result = self.client.query(schema_query)
            
            # Build schema dictionary
            schema = {}
            for row in result.result_rows:
                name, type_, default = row[0], row[1], row[3] if len(row) > 3 else None
                schema[name] = {
                    'type': type_,
                    'default': default
                }
            
            self.schema_cache[table_name] = schema
            return schema
            
        except Exception as e:
            logger.error(f"Error getting schema for table {table_name}: {str(e)}")
            return {}

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
            
            if not table_result:
                return None
                
            row = table_result[0]
            stats = {
                'name': row['table'],
                'total_rows': row['total_rows'],
                'total_bytes': row['total_bytes'],
                'engine': row['engine'],
                'partition_key': row['partition_key'],
                'sorting_key': row['sorting_key'],
                'primary_key': row['primary_key'],
                'sampling_key': row['sampling_key'],
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
                    if stats_result:
                        row = stats_result[0]
                        for i, col in enumerate(numeric_columns):
                            base_idx = i * 5  # 5 stats per column
                            stats['column_stats'][col] = {
                                'min': row[f'min_{col}'] if row[f'min_{col}'] is not None else 0,
                                'max': row[f'max_{col}'] if row[f'max_{col}'] is not None else 0,
                                'avg': row[f'avg_{col}'] if row[f'avg_{col}'] is not None else 0,
                                'count': row[f'count_{col}'] if row[f'count_{col}'] is not None else 0,
                                'unique_count': row[f'unique_{col}'] if row[f'unique_{col}'] is not None else 0
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
            
        # Add LIMIT if not present
        if 'LIMIT' not in query.upper():
            query += ' LIMIT 1000'
            
        return query

    def generate_query(self, question: str) -> str:
        """Generate a SQL query based on the question using knowledge base"""
        try:
            # Load the knowledge base first
            with open('product_demand_knowledge.json', 'r') as f:
                knowledge_base = json.load(f)
            
            # Get the schema from ClickHouse
            schema = self.get_table_schema('ProductWiseDemandSales')
            
            # Create the system prompt using the knowledge base
            system_prompt = f"""You are an AI data analyst specialized in analyzing product demand and sales data.
            
            IMPORTANT KNOWLEDGE BASE RULES:
            1. Date Handling:
            {json.dumps(knowledge_base['product_demand_sales']['business_rules']['date_handling'], indent=2)}
            
            2. Value Calculations:
            {json.dumps(knowledge_base['product_demand_sales']['business_rules']['value_calculation_rules'], indent=2)}
            
            3. Analysis Patterns:
            {json.dumps(knowledge_base['product_demand_sales']['analysis_patterns'], indent=2)}
            
            QUERY GUIDELINES:
            1. Use only these fields and follow their exact usage:
            {json.dumps(knowledge_base['product_demand_sales']['table_structure'], indent=2)}
            
            2. Follow these example patterns:
            {json.dumps(knowledge_base['product_demand_sales']['query_examples'], indent=2)}
            
            3. Additional Rules:
            - ALWAYS use the exact table name: ProductWiseDemandSales
            - NEVER use any other table name (like 'sales')
            - Only use existing table columns from this schema: {json.dumps(schema)}
            - Include proper WHERE clauses
            - Use appropriate aggregations
            - Do not include semicolons or comments
            - Ensure queries are performant
            - Always alias columns with meaningful names
            - ALWAYS start queries with SELECT
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}\nAvailable Columns: {', '.join(schema.keys())}"}
            ]
            
            response = openai.ChatCompletion.create(
                engine=OPENAI_DEPLOYMENT,
                messages=messages,
                temperature=0.1,
                max_tokens=1000
            )
            
            query = response.choices[0].message.content.strip()
            
            # Clean and optimize the query
            query = self.clean_sql_query(query)
            query = self.optimize_query(query)
            
            logger.info(f"Generated query: {query}")
            return query
            
        except Exception as e:
            logger.error(f"Error generating query: {str(e)}")
            raise

    def optimize_query(self, query: str) -> str:
        """Optimize the generated query for ClickHouse."""
        try:
            # Add FINAL modifier for tables with ReplacingMergeTree engine
            tables_query = "SELECT name, engine FROM system.tables WHERE database = currentDatabase()"
            tables_result = self.execute_query(tables_query, "Get table engines")
            
            for row in tables_result:
                table, engine = row['name'], row['engine']
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
            return "\n".join(row['explain'] for row in result)
        except Exception as e:
            logger.error(f"Error explaining query: {e}")
            return ""

@app.post("/analyze")
async def analyze_data(query: Query):
    """Analyze data based on the question"""
    try:
        start_time = datetime.now()
        logger.info(f"Starting analysis for question: {query.question}")
        
        # Initialize query generator
        generator = ClickHouseQueryGenerator()
        
        # Generate and optimize query
        logger.info("Generating SQL query...")
        sql_query = generator.generate_query(query.question)
        logger.info(f"Generated query: {sql_query}")
        
        optimized_query = generator.optimize_query(sql_query)
        logger.info(f"Optimized query: {optimized_query}")
        
        # Execute query
        logger.info("Executing query...")
        results = generator.execute_query(optimized_query, "Analyzing data based on question")
        logger.info(f"Query execution complete. Results: {results[:100] if results else 'No results'}")
        
        # Calculate performance metrics
        end_time = datetime.now()
        performance_metrics = {
            "execution_time_ms": (end_time - start_time).total_seconds() * 1000,
            "query_complexity": len(sql_query.split()),
            "metrics_used": [col for col in results[0].keys()] if results else []
        }
        
        # Learn from this interaction
        logger.info("Learning from interaction...")
        generator.learn_from_interaction(
            question=query.question,
            generated_query=optimized_query,
            results=results,
            performance_metrics=performance_metrics
        )
        
        response_data = {
            "query": optimized_query,
            "results": results,
            "performance": performance_metrics
        }
        
        logger.info("Analysis complete")
        return response_data
        
    except Exception as e:
        error_msg = f"Error in analyze_data: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
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
        
        for row in tables_result:
            table_name = row['table']
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
        tables = [row['name'] for row in tables_result]
        
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
