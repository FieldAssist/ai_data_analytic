# AI Data Analyst System Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Backend Components](#backend-components)
4. [Frontend Components](#frontend-components)
5. [Knowledge Base System](#knowledge-base-system)
6. [Setup and Installation](#setup-and-installation)
7. [API Reference](#api-reference)

## System Overview

The AI Data Analyst is an intelligent system that combines natural language processing, machine learning, and data visualization to provide intuitive data analysis capabilities. It allows users to query complex data using natural language and automatically generates optimized SQL queries for ClickHouse database.

### Key Features
- Natural language to SQL conversion
- Intelligent query optimization
- Interactive data visualization
- Continuous learning system
- Real-time performance monitoring

## Architecture

### Tech Stack
- **Backend**: Python FastAPI
- **Frontend**: Vue.js 3.x
- **Database**: ClickHouse
- **AI Service**: Azure OpenAI
- **Visualization**: Chart.js

### System Components
```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│    Frontend     │────▶│     Backend      │────▶│   ClickHouse   │
│    (Vue.js)     │◀────│    (FastAPI)     │◀────│   Database     │
└─────────────────┘     └──────────────────┘     └────────────────┘
                               │    ▲
                               ▼    │
                        ┌──────────────────┐
                        │   Knowledge Base │
                        │     System      │
                        └──────────────────┘
```

## Backend Components

### Query Generation System (`ClickHouseQueryGenerator`)

#### Core Functions
1. **Query Generation**
   ```python
   def generate_query(question: str) -> str:
       # Converts natural language to SQL
       # Uses AI to understand context
       # Returns optimized SQL query
   ```

2. **Query Optimization**
   ```python
   def optimize_query(query: str) -> str:
       # Applies performance optimizations
       # Adds necessary limits and filters
       # Ensures ClickHouse compatibility
   ```

3. **Query Execution**
   ```python
   def execute_query(query: str) -> Dict:
       # Executes the query safely
       # Handles data type conversions
       # Returns formatted results
   ```

### API Endpoints

#### Main Endpoints
- `POST /analyze`
  - Processes natural language queries
  - Returns query results and visualizations
  - Includes performance metrics

- `GET /tables`
  - Returns database schema information
  - Includes table statistics
  - Shows available columns and types

- `POST /analyze_query`
  - Analyzes SQL queries without execution
  - Provides query optimization suggestions
  - Shows estimated performance metrics

## Frontend Components

### Main Application (`App.vue`)
- Handles user interactions
- Manages application state
- Coordinates component communication

### Data Visualization Components

#### DataChart.vue
```javascript
// Supports multiple chart types
- Bar charts
- Line charts
- Area charts
- Scatter plots
```

#### DataVisualization.vue
- Dynamic visualization selection
- Responsive design
- Interactive data exploration

### User Interface Features
1. Query Input
   - Natural language input field
   - Query history
   - Suggestion system

2. Results Display
   - Interactive charts
   - Data tables
   - Export options

3. Performance Metrics
   - Query execution time
   - Data volume metrics
   - Optimization suggestions

## Knowledge Base System

### Storage Schema
```sql
CREATE TABLE knowledge_base (
    timestamp DateTime,
    knowledge_id String,
    raw_knowledge String,
    analysis String,
    patterns String
) ENGINE = MergeTree()
ORDER BY (timestamp, knowledge_id)
```

### Learning Process
1. **Pattern Recognition**
   - Query pattern analysis
   - Data relationship identification
   - Business rule extraction

2. **Knowledge Application**
   - Pattern matching
   - Query optimization
   - Result enhancement

3. **Continuous Learning**
   - User interaction analysis
   - Performance optimization
   - Pattern refinement

## Setup and Installation

### Prerequisites
- Python 3.8+
- Node.js 14+
- ClickHouse Database
- Azure OpenAI API access

### Backend Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your configurations

# Start the backend server
uvicorn app.main:app --reload
```

### Frontend Setup
```bash
# Install dependencies
cd vue-frontend
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## API Reference

### Query Analysis
```http
POST /analyze
Content-Type: application/json

{
    "question": "Show me total sales by product for last month"
}
```

### Table Information
```http
GET /tables
```

### Query Analysis
```http
POST /analyze_query
Content-Type: application/json

{
    "query": "SELECT * FROM sales LIMIT 100"
}
```

## Performance Optimization

### Query Optimization Techniques
1. Automatic limit addition
2. Index utilization
3. Join optimization
4. Aggregation optimization

### Caching Strategy
1. Schema caching
2. Query result caching
3. Knowledge base caching

## Security Considerations

### Data Access
- Role-based access control
- Query validation
- Input sanitization

### API Security
- Authentication
- Rate limiting
- CORS configuration

## Troubleshooting

### Common Issues
1. Connection Issues
   - Check database connectivity
   - Verify API credentials
   - Check network settings

2. Performance Issues
   - Monitor query complexity
   - Check data volume
   - Review cache usage

3. Query Generation Issues
   - Verify knowledge base
   - Check question formatting
   - Review error logs

## Contributing

### Development Guidelines
1. Code Style
   - Follow PEP 8 for Python
   - Use Vue.js style guide
   - Document all functions

2. Testing
   - Write unit tests
   - Perform integration testing
   - Test with sample data

3. Documentation
   - Update API documentation
   - Document new features
   - Maintain changelog

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*Last updated: February 25, 2025*
