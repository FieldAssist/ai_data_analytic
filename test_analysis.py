import requests
import json

def analyze_question(question: str):
    response = requests.post(
        'http://localhost:8000/analyze',
        json={'question': question}
    )
    
    print(f"\nQuestion: {question}")
    print("\nResponse:")
    print(json.dumps(response.json(), indent=2))

# Test with a specific question
question = "What are the top 5 products by invoice value for November 2024 for company 10830?"
analyze_question(question)
