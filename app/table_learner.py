import json
import logging
from typing import Dict, Any, List, Optional
import openai
from datetime import datetime
from clickhouse_connect import get_client

logger = logging.getLogger(__name__)

class TableLearner:
    def __init__(self):
        self.client = get_client(
            host='20.235.209.193',
            port=8123,
            username='admin',
            password='2286vdaC8LN94RmdTrctyXZPavHcx8',
            database='unify'
        )
        self._initialize_storage()
        
    def _initialize_storage(self) -> None:
        """Initialize the knowledge base storage in ClickHouse"""
        try:
            # Create table for knowledge base if it doesn't exist
            self.client.command("""
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

    def learn_from_knowledge(self, knowledge_data: Dict[str, Any]) -> None:
        """
        Learn patterns and insights from provided knowledge data
        """
        try:
            # Generate unique ID for this knowledge entry
            timestamp = datetime.now()
            knowledge_id = f"k_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Use Azure OpenAI to analyze patterns
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
                engine=openai.api_type,
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            
            # Extract patterns
            analysis = response.choices[0].message.content
            patterns = self._extract_patterns(analysis)
            
            # Store in ClickHouse
            self.client.command("""
                INSERT INTO knowledge_base (
                    timestamp, knowledge_id, raw_knowledge, analysis, patterns
                ) VALUES
                """, parameters={
                    'timestamp': timestamp,
                    'knowledge_id': knowledge_id,
                    'raw_knowledge': json.dumps(knowledge_data),
                    'analysis': analysis,
                    'patterns': json.dumps(patterns)
                })
            
            logger.info(f"Successfully learned and stored knowledge. ID: {knowledge_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error learning from knowledge: {str(e)}")
            raise
    
    def _extract_patterns(self, analysis: str) -> Dict[str, Any]:
        """Extract structured patterns from the analysis"""
        try:
            # Use Azure OpenAI to structure the analysis
            messages = [
                {"role": "system", "content": "Convert the analysis into a structured JSON format with patterns, rules, and metrics."},
                {"role": "user", "content": analysis}
            ]
            
            response = openai.ChatCompletion.create(
                engine=openai.api_type,
                messages=messages,
                temperature=0.1,
                max_tokens=1000
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error extracting patterns: {str(e)}")
            return {"error": str(e)}
    
    def get_relevant_knowledge(self, query: str) -> Dict[str, Any]:
        """
        Retrieve relevant knowledge based on a query
        """
        try:
            # Get all knowledge from ClickHouse
            result = self.client.query("""
                SELECT timestamp, knowledge_id, raw_knowledge, analysis, patterns
                FROM knowledge_base
                ORDER BY timestamp DESC
            """)
            
            if not result.result_rows:
                return {"message": "No knowledge has been learned yet"}
            
            # Convert to dictionary format
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
            context = json.dumps(knowledge_base)
            messages = [
                {"role": "system", "content": "Find and return the most relevant knowledge patterns for the given query."},
                {"role": "user", "content": f"Query: {query}\nKnowledge Base: {context}"}
            ]
            
            response = openai.ChatCompletion.create(
                engine=openai.api_type,
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error retrieving relevant knowledge: {str(e)}")
            return {"error": str(e)}
    
    def apply_learned_knowledge(self, query: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply learned knowledge to enhance query analysis
        """
        try:
            relevant_knowledge = self.get_relevant_knowledge(query)
            
            # Use Azure OpenAI to apply knowledge to the data
            messages = [
                {"role": "system", "content": "Apply the relevant knowledge patterns to analyze the data and provide insights."},
                {"role": "user", "content": f"Query: {query}\nData: {json.dumps(data)}\nRelevant Knowledge: {json.dumps(relevant_knowledge)}"}
            ]
            
            response = openai.ChatCompletion.create(
                engine=openai.api_type,
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error applying learned knowledge: {str(e)}")
            return {"error": str(e)}
