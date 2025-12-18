import requests
import json

# Test the API endpoint
url = "http://localhost:5000/api/predict"

test_data = {
    "disaster_type": "Earthquake",
    "location": "Japan",
    "latitude": 35.6762,
    "longitude": 139.6503,
    "month": 7,
    "week": 2
}

print("Testing API endpoint...")
print(f"URL: {url}")
print(f"Data: {json.dumps(test_data, indent=2)}\n")

try:
    response = requests.post(url, json=test_data)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n✓ API Response:")
        print(json.dumps(result, indent=2))
    else:
        print(f"\n✗ Error: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("\n✗ Error: Cannot connect to Flask server")
    print("   Make sure the Flask app is running at http://localhost:5000")
except Exception as e:
    print(f"\n✗ Error: {e}")
