import requests
import json

def test_analyze():
    url = "http://localhost:8000/analyze"
    data = {
        "question": "Show me all tables"
    }
    
    print("Sending request to:", url)
    print("Request data:", json.dumps(data, indent=2))
    
    try:
        response = requests.post(url, json=data)
        print("\nResponse status:", response.status_code)
        print("Response headers:", dict(response.headers))
        print("\nResponse body:", json.dumps(response.json(), indent=2))
    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    test_analyze()
