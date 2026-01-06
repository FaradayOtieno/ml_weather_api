import requests

base_url = "http://127.0.0.1:8000/weather_predict/"
headers = {"x-api-key": "secret-token-123"}
cities = ["Mombasa", "Nairobi", "Kisumu", "Garissa"]

for city in cities:
    response = requests.get(f"{base_url}{city}", headers=headers)
    if response.status_code == 200:
        data = response.json()
        print(f"{city} → Temp: {data['temperature']}°C, Weather: {data['weather']}, ML Prediction: {data['ml_prediction']}")
    else:
        print(f"{city} → Error: {response.status_code}, {response.json()['detail']}")
