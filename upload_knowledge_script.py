import json
import requests

# Read the knowledge file
with open('product_demand_knowledge.json', 'r') as f:
    knowledge_data = json.load(f)

# Upload to the knowledge base
response = requests.post(
    'http://localhost:8001/upload_knowledge',
    json={'knowledge': knowledge_data}
)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")
