import json
import logging
from datetime import datetime
from clickhouse_connect import get_client
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI
openai.api_type = "azure"
openai.api_version = "2024-02-15-preview"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-35-turbo")

def get_clickhouse_client():
    """Get configured ClickHouse client"""
    return get_client(
        host='20.235.209.193',
        port=8123,
        username='admin',
        password='2286vdaC8LN94RmdTrctyXZPavHcx8',
        database='unify'
    )

def initialize_storage(client):
    """Initialize the knowledge base storage in ClickHouse"""
    try:
        # Create table for knowledge base if it doesn't exist
        client.command("""
            CREATE TABLE IF NOT EXISTS knowledge_base (
                timestamp DateTime,
                knowledge_id String,
                raw_knowledge String,
                analysis String,
                patterns String
            ) ENGINE = MergeTree()
            ORDER BY (timestamp, knowledge_id)
        """)
        logger.info("Knowledge base storage initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing storage: {str(e)}")
        raise

def analyze_knowledge(knowledge_data):
    """Analyze knowledge using Azure OpenAI"""
    system_prompt = """Analyze the provided knowledge data and extract:
    1. Key patterns and relationships
    2. Business rules and constraints
    3. Common data transformations
    4. Relevant metrics and KPIs
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Analyze this knowledge: {json.dumps(knowledge_data)}"}
    ]
    
    response = openai.ChatCompletion.create(
        engine=OPENAI_DEPLOYMENT,
        messages=messages,
        temperature=0.3,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

def extract_patterns(analysis):
    """Extract structured patterns from the analysis"""
    messages = [
        {"role": "system", "content": "Convert the analysis into a structured JSON format with patterns, rules, and metrics."},
        {"role": "user", "content": analysis}
    ]
    
    response = openai.ChatCompletion.create(
        engine=OPENAI_DEPLOYMENT,
        messages=messages,
        temperature=0.1,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

def main():
    try:
        # Initialize ClickHouse client
        client = get_clickhouse_client()
        
        # Initialize storage
        initialize_storage(client)
        
        # Read knowledge file
        with open('product_demand_knowledge.json', 'r') as f:
            knowledge_data = json.load(f)
        
        # Generate unique ID
        timestamp = datetime.now()
        knowledge_id = f"k_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Analyze knowledge
        logger.info("Analyzing knowledge...")
        analysis = analyze_knowledge(knowledge_data)
        
        # Extract patterns
        logger.info("Extracting patterns...")
        patterns = extract_patterns(analysis)
        
        # Store in ClickHouse
        logger.info("Storing in ClickHouse...")
        client.command("""
            INSERT INTO knowledge_base (
                timestamp, knowledge_id, raw_knowledge, analysis, patterns
            ) VALUES (
                %(timestamp)s, %(knowledge_id)s, %(raw_knowledge)s, %(analysis)s, %(patterns)s
            )
            """, parameters={
                'timestamp': timestamp,
                'knowledge_id': knowledge_id,
                'raw_knowledge': json.dumps(knowledge_data),
                'analysis': analysis,
                'patterns': patterns
            })
        
        logger.info(f"Successfully uploaded knowledge with ID: {knowledge_id}")
        
        # Verify storage
        result = client.query("""
            SELECT timestamp, knowledge_id, analysis
            FROM knowledge_base
            WHERE knowledge_id = %(knowledge_id)s
            LIMIT 1
        """, parameters={'knowledge_id': knowledge_id})
        
        if result.result_rows:
            logger.info("Successfully verified knowledge storage")
            logger.info(f"Stored analysis: {result.result_rows[0][2][:200]}...")
        
    except Exception as e:
        logger.error(f"Error uploading knowledge: {str(e)}")
        raise

if __name__ == "__main__":
    main()
