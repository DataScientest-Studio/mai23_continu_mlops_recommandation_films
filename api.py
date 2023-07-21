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
    version = "0.0.1",
    openapi_tags = [
        {"name": "Home",
         "description": "This is the Home route"},
        {"name": "new_user",
         "description": "This is the new_user route"},
        {"name": "new_rating",
         "description": "This is the new_rating route"},
        {"name": "recommendation_system",
         "description": "This is the Recommendation System route"},
        {"name": "log_event",
         "description": "This is the log_event route"}])



@api.get('/', tags = ["home"]) # default route
def get_home():
    """
    This is the home route
    """
    return {"Welcome to our API. This is a work in progress."}



@api.put("/new_user/{user: User}", tags = ["new_user"])
def new_user(user):
    """
    This is the new_user route
    """
    return {"This will add a new user."}
        
    

@api.get("/new_rating/{rating: Rating}", tags = ["new_rating"])
def new_rating(rating):
    """
    This is the new_rating route
    """
    return {"This will add a new rating"}



@api.post("/recommendation_system/{credentials: Credentials}", tags = ["recommendation"])
def recommendation_system(credentials):
    """
    This is the recommendation_system route
    """
    return hybrid_recommendation_movies(credentials.user_id, credentials.movie_id)



@api.post("/log_event/{event: Event}", tags = ["log_event"])
def log_event(event):
    return {"This is the log_event route"}




