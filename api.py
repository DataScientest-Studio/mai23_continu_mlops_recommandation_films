import pandas as pd
import numpy as np  
from fastapi import FastAPI
from classes import User, Credentials, Rating, Event
from api_recommendation import hybrid_recommendation_movies
import sqlite3


#with open("model.pkl", "rb") as pickled:
#   model = pickle.load(pickled)


api = FastAPI(
    title = "API project Recommnendation System, MLOps May 2023",
    description = "This is a Recommendation System API",
    version = "0.1.1",
    openapi_tags = [
        {"name": "home",
         "description": "This is the Home route"},
        {"name": "new_user",
         "description": "This is the new_user route"},
        {"name": "new_rating",
         "description": "This is the new_rating route"},
        {"name": "recommendation_system",
         "description": "This is the Recommendation System route"},
        {"name": "log_event",
         "description": "This is the log_event route"}])


def connect_to_db(db):
    connection = sqlite3.connect(db)
    connection.row_factory = sqlite3.Row
    return connection


@api.get('/', tags = ["home"]) # default route
def get_home():
    """
    This is the home route
    """
    return {"Welcome to our API. This is a work in progress."}




@api.post("/new_user", tags = ["new_user"])
def new_user(user: User):
    """
    This is the new_user route
    """
    conn = connect_to_db("database.db")
    cursor = conn.cursor()
    try:
        cursor.execute(f"INSERT INTO users (name, email, password) VALUES (?,?,?)", (user.name, user.email, user.password.get_secret_value()))
        new_user_id = cursor.lastrowid
        conn.commit()
        return {"userid": new_user_id}
    except sqlite3.IntegrityError:
        print("User already exists")
    except sqlite3.ProgrammingError:
        print("SQL syntax error")
    except sqlite3.OperationalError:
        print("Operational issue")
    except sqlite3.DatabaseError:
        print("Database error")
    conn.close()
    return {None}
    
    
@api.delete("/delete_user", tags = ["delete_user"])
def delete_user(user: User):
    """
    This is the delete_user route
    """
    conn = connect_to_db("database.db")
    cursor = conn.cursor()
    success = False
    try:
        cursor.execute(f"DELETE FROM users WHERE userid = ?", (user.userid,))
        conn.commit()
        success = True
    except sqlite3.OperationalError:
        print("Operational issue")
    except sqlite3.DatabaseError:
        print("Database error")
    conn.close()
    return {f"Success: {success}"}


@api.patch("update_user", tags = ["update_user"])
def update_user(user: User):
    return {f"When this route grows up it will update the user: {user}"}


@api.post("/new_rating", tags = ["new_rating"])
def new_rating(rating: Rating):
    """
    This is the new_rating route
    """
    return {f"When this route grows up it will add the new rating: {rating}"}


@api.post("/recommendation_system", tags = ["recommendation_system"])
async def recommendation_system(userid : int, movie : str):
    """
    This is the recommendation_system route.

    input : 
    
        userid : integer
        movie_title : string

    Return movies from a recommendation system
    """

    recommendation_movies = hybrid_recommendation_movies(userid,movie)
    
    return {f"When this route grows up it will provide recommendations for this movie: {movie}" : recommendation_movies}


@api.post("/log_event", tags = ["log_event"])
def log_event(event: Event):
    return {"This is the log_event route"}




