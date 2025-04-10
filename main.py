#import kagglehub
# Download latest version
#path = kagglehub.dataset_download("akshatgupta7/crop-yield-in-indian-states-dataset")
#print("Path to dataset files:", path)

#import pandas as pd
#dataset_path = r"C:\Users\daksh\.cache\kagglehub\datasets\akshatgupta7\crop-yield-in-indian-states-dataset\versions\1\Crop_Yield.csv"
#df = pd.read_csv(dataset_path)
#print(df.head())

#from fastapi import FastAPI

#app = FastAPI()

#@app.get("/")
#def read_root():
   # return {"message": "Crop & Veggie Yield Prediction API is running!"}
# VERSION 2
#from fastapi import FastAPI
#from pydantic import BaseModel
#from fastapi.responses import JSONResponse

#app = FastAPI(
   # title="Crop & Vegetable Yield Prediction API",
  #  description="A simple API to predict agricultural yield using ML models",
 #   version="1.0.0"
#)

# Root route
#@app.get("/")
#def read_root():
 #   return {"message": "Welcome to the Crop & Veggie Yield Prediction API!"}

# Health check
#@app.get("/health")
#def health_check():
 #   return JSONResponse(content={"status": "ok"}, status_code=200)

# Input data schema
#lass YieldInput(BaseModel):
    #state: str
    #district: str
    #season: str
   # crop: str
  #  rainfall: float
 #   temperature: float

# Prediction route
#@app.post("/predict")
#def predict_yield(data: YieldInput):
    # Placeholder for now – replace with actual model call
    #dummy_prediction = 2750.5  # dummy prediction in kg/ha

   # return {
  #      "input_data": data.dict(),
 #       "predicted_yield_kg_per_ha": dummy_prediction
#    }

#VERSION 3
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import numpy as np
import pickle

app = FastAPI(
    title="Crop & Vegetable Yield Prediction API",
    description="A simple API to predict agricultural yield using ML models",
    version="1.0.0"
)

# Load saved model, scaler, and encoders
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Health check
@app.get("/health")
def health_check():
    return JSONResponse(content={"status": "ok"}, status_code=200)

# Root route
@app.get("/")
def read_root():
    return {"message": "Welcome to the Crop & Veggie Yield Prediction API!"}

# Input schema
class YieldInput(BaseModel):
    State: str
    Crop_Year: int
    Crop: str
    Season: str
    Area: float
    Annual_Rainfall: float
    Fertilizer: float
    Pesticide: float

# Prediction route
@app.post("/predict")
def predict_yield(data: YieldInput):
    try:
        # Encode categorical values
        state_encoded = label_encoders['State'].transform([data.State])[0]
        crop_encoded = label_encoders['Crop'].transform([data.Crop])[0]
        season_encoded = label_encoders['Season'].transform([data.Season])[0]

        # Construct input array
        input_data = np.array([[state_encoded, data.Crop_Year, crop_encoded, season_encoded,
                                data.Area, data.Annual_Rainfall, data.Fertilizer, data.Pesticide]])

        # Scale the input
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)[0]

        return {
            "input_data": data.dict(),
            "predicted_yield_kg": round(prediction, 2)
        }

    except Exception as e:
        return {"error": str(e)}
