import sys
import os
# Ensure current folder is in Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Header, Depends
import logging
import requests
import torch
from ml_model import MLP
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
ML_API_TOKEN = os.getenv("ML_API_TOKEN")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not ML_API_TOKEN or not OPENWEATHER_API_KEY:
    raise ValueError("API tokens not found in .env file")

# Logging
logging.basicConfig(level=logging.INFO)

# FastAPI app
app = FastAPI(title="ML + Weather Demo API")

# Token authentication
def token_auth(x_api_key: str = Header(...)):
    if x_api_key != ML_API_TOKEN:
        logging.warning(f"Unauthorized token attempt: {x_api_key}")
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

# Load ML model
model_path = "models/mlp_model.pth"
model = MLP()
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

@app.get("/weather_predict/{city}")
def weather_predict(city: str, auth: bool = Depends(token_auth)):
    # Fetch weather
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    res = requests.get(url)
    if res.status_code != 200:
        raise HTTPException(status_code=404, detail="City not found or API error")
    
    data = res.json()
    temp = data['main']['temp']
    weather_desc = data['weather'][0]['description']

    # ML prediction
    x_tensor = torch.tensor([[temp]], dtype=torch.float32)
    with torch.no_grad():
        out = model(x_tensor)
        pred = torch.argmax(out, dim=1).item()
    
    logging.info(f"{city}: Temp={temp}°C, ML Prediction={pred}")
    
    return {
        "city": city,
        "temperature": temp,
        "weather": weather_desc,
        "ml_prediction": pred
    }

@app.get("/")
def root():
    return {"message": "ML + Weather Demo API running. Use /weather_predict/{city} with x-api-key header."}