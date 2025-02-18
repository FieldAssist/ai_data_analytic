# AI Data Analyst

An intelligent platform that allows users to analyze data through natural language queries, generating instant charts and insights from SQL and ClickHouse databases.

## Features

- Natural language to SQL query conversion
- Interactive data visualization
- Support for both SQL and ClickHouse databases
- Real-time chart generation
- Intelligent data insights

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your database credentials:
```
SQL_DATABASE_URL=your_sql_database_url
CLICKHOUSE_HOST=your_clickhouse_host
CLICKHOUSE_PORT=your_clickhouse_port
CLICKHOUSE_USER=your_clickhouse_user
CLICKHOUSE_PASSWORD=your_clickhouse_password
OPENAI_API_KEY=your_openai_api_key
```

4. Run the application:
```bash
uvicorn app.main:app --reload
```

## Project Structure

- `/app`: Main application directory
  - `main.py`: FastAPI application entry point
  - `database.py`: Database connection handlers
  - `models.py`: Database models
  - `schemas.py`: Pydantic models for request/response
  - `ai_service.py`: AI query processing service
  - `chart_service.py`: Chart generation service
- `/frontend`: React frontend application
