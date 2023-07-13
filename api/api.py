import pandas as pd
import numpy as np  
import pickle 
# import uvicorn
from fastapi import FastAPI
from classes import User


with open("model.pkl", "rb") as pickled:
    model = pickle.load(pickled)

app = FastAPI()

@app.get('/') # default route
def get_index():
    return {"This is the default route"}

@app.get("/generate_recommendations")
def generate():
    return {"This is the generate_recommendations route"}

app.post("/log_event")
def log_event():
    return {"This is the log_event route"}
