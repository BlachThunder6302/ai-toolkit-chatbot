import requests
import json
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get API key from environment
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

data = {
    "model": "mistralai/mistral-7b-instruct:free",
    "temperature": 0.7,
    "max_tokens": 150,
    "messages": [
        {"role": "system", "content": "You are a helpful and detailed AI assistant."},
        {"role": "user", "content": "Give me three different philosophical interpretations of the meaning of life."}
    ]
}

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers=headers,
    json=data,
)

print("Status:", response.status_code)
result = response.json()
print(json.dumps(result, indent=2))

if "choices" in result and len(result["choices"]) > 0:
    content = result["choices"][0]["message"].get("content", "").strip()
    if content:
        print("\nAssistant reply:\n", content)
    else:
        print("\n⚠️ The model returned an empty message.")
else:
    print("\n⚠️ Unexpected response structure.")
