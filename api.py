import pandas as pd
import numpy as np  
import pickle 
# import uvicorn
from fastapi import FastAPI
from classes import User
from recommendation_system import hybrid_recommendation_movies


#with open("model.pkl", "rb") as pickled:
#   model = pickle.load(pickled)

api = FastAPI(
    title = "API project Recommnendation System, MLOps May 2023",
    description = "This is a Recommendation System API",
    version = "0.0.1"
    )

@api.get('/') # default route
def get_index():
    return {"Welcome to our API. This is a work in progress."}

@api.get("/add_user/{user: User}")
def add_user(user):
    return {"This is the add_user route"}
        
    

#@api.get("/get_recommendations")


@api.get("/generate_recommendations")
def generate():
    return {"This is the generate_recommendations route"}

@api.post("/log_event")
def log_event():
    return {"This is the log_event route"}
