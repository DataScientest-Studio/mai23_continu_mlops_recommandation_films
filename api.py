import pandas as pd
import numpy as np  
import pickle 
# import uvicorn
from fastapi import FastAPI
from classes import User, Credentials, Rating, Event
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



@api.put("/new_user", tags = ["new_user"])
def new_user(user: User):
    """
    This is the new_user route
    """
    return {"This will add a new user."}
        
    

@api.post("/new_rating/", tags = ["new_rating"])
def new_rating(rating: Rating):
    """
    This is the new_rating route
    """
    return {"This will add a new rating"}



@api.post("/recommendation_system", tags = ["recommendation"])
def recommendation_system(credentials: Credentials, movieid, ):
    """
    This is the recommendation_system route
    """
    return hybrid_recommendation_movies(credentials, movieid, )



@api.post("/log_event", tags = ["log_event"])
def log_event(event: Event):
    return {"This is the log_event route"}




