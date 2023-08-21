import pandas as pd
import numpy as np  
from fastapi import FastAPI
from classes import User, Event, Rating
#from api_recommendation import hybrid_recommendation_movies
import sqlite3


#with open("model.pkl", "rb") as pickled:
#   model = pickle.load(pickled)


api = FastAPI(
    title = "API project Recommnendation System, MLOps May 2023",
    description = "This is a Recommendation System API",
    version = "0.5.1",
    version_detail  = "1. addition of routes: login, get_user, delete_rating (for specific userid+movieid+date combo)\
                        2. check_password function (for login)",
    openapi_tags = [
        {"name": "home",
         "description": "This is the Home route"},
        {"name": "new_user",
         "description": "This is the new_user route"},
        {"name": "delete_user",
         "description": "This is the delete_user route"},
        {"name": "update_user",
         "description": "This is the update_user route"},
        {"name": "new_rating",
         "description": "This is the new_rating route"},
        {"name": "delete_ratings",
         "description": "This is the delete_ratings route"},
        {"name": "update_rating",
         "description": "This is the update_rating route"},
        {"name": "recommendation_system",
         "description": "This is the Recommendation System route"},
        {"name": "log_event",
         "description": "This is the log_event route"},]
)


def connect_to_db(db):
    connection = sqlite3.connect(db)
    connection.row_factory = sqlite3.Row
    return connection


def check_password(user:User):
    conn = connect_to_db("database.db")
    cursor = conn.cursor()
    password = None
    try:
        cursor.execute(f"SELECT * FROM users WHERE userid = ?", (user.userid,))
        password = cursor.fetchone()
        #print(password["password"])
    except sqlite3.OperationalError:
        print("Operational issue")
    except sqlite3.DatabaseError:
        print("Database error")
    conn.close()
    check = password == user.password.get_secret_value()
    return check  

  
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
    success = False
    try:
        cursor.execute(f"INSERT INTO users (name, email, password) VALUES (?,?,?)", (user.name, user.email, user.password.get_secret_value()))
        new_user_id = cursor.lastrowid
        conn.commit()
        success = True
        return {"Success": success,"userid": new_user_id}
    except sqlite3.IntegrityError:
        print("User already exists")
    except sqlite3.ProgrammingError:
        print("SQL syntax error")
    except sqlite3.OperationalError:
        print("Operational issue")
    except sqlite3.DatabaseError:
        print("Database error")
    conn.close()
    return {"Success": success}


@api.put("/login", tags = ["login"])
def login(user: User):
    print(user)
    success = False
    success = check_password(user)
    return {"Success": success}


@api.get("/get_user", tags = ["get_user"])
def get_user(userid: int):
    """
    This is the get_user route
    """
    conn = connect_to_db("database.db")
    cursor = conn.cursor()
    user_data = None
    user = None
    success = False
    try:
        cursor.execute(f"SELECT * FROM users WHERE userid = ?", (userid,))
        user_data = cursor.fetchone()
        #print(user_data["name"], user_data["email"], user_data["password"])
        user = User(userid = user_data["userid"], name = user_data["name"], email = user_data["email"], password = user_data["password"])
        success = True
    except sqlite3.OperationalError:
        print("Operational issue")
    except sqlite3.DatabaseError:
        print("Database error")
    conn.close()
    return {"Success": success, "User": user}
    
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
        cursor.execute(f"DELETE FROM ratings WHERE userid = ?", (user.userid,))
        conn.commit()
        success = True
    except sqlite3.OperationalError:
        print("Operational issue")
    except sqlite3.DatabaseError:
        print("Database error")
    conn.close()
    return {"Success": success}


@api.patch("/update_user/", tags = ["update_user"])
def update_user(update: User, field: str):
    """
    This is the update_user route
    """
    conn = connect_to_db("database.db")
    cursor = conn.cursor()
    success = False
    value = ""
    if field == "name":
        value = update.name
        try:
            cursor.execute(f"UPDATE users SET name = ? WHERE userid = ?", (value, update.userid))
            conn.commit()
            success = True
        except sqlite3.OperationalError:
            print("Operational issue")
        except sqlite3.DatabaseError:
            print("Database error")
    if field == "email":
        value = update.email
        try:
            cursor.execute(f"UPDATE users SET email = ? WHERE userid = ?", (value, update.userid))
            conn.commit()
            success = True
        except sqlite3.OperationalError:
            print("Operational issue")
        except sqlite3.DatabaseError:
            print("Database error")
    if field == "password":
        value = update.password.get_secret_value()
        try:
            cursor.execute(f"UPDATE users SET password = ? WHERE userid = ?", (value, update.userid))
            conn.commit()
            success = True
        except sqlite3.OperationalError:
            print("Operational issue")
        except sqlite3.DatabaseError:
            print("Database error")
    conn.close()
    return {"Success": success}


@api.post("/new_rating", tags = ["new_rating"])
def new_rating(rating: Rating):
    """
    This is the new_rating route
    
    """
    conn = connect_to_db("database.db")
    cursor = conn.cursor()
    success = False
    try:
        cursor.execute(f"INSERT INTO ratings (userid, movieid, date, rating) VALUES (?,?,?,?)", (rating.userid, rating.movieid, rating.date, rating.rating))
        # new_rating_id = cursor.lastrowid REMOVE WHEN SURE NO LONGER NEEDED
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        print("Rating already exists for this film by this user.")
    except sqlite3.ProgrammingError:
        print("SQL syntax error")
    except sqlite3.OperationalError:
        print("Operational issue")
    except sqlite3.DatabaseError:
        print("Database error")
    conn.close()
    return {"Success": success}


@api.delete("/delete_rating", tags = ["delete_rating"])
def delete_rating(rating: Rating):
    """
    This is the delete_rating route
    """
    conn = connect_to_db("database.db")
    cursor = conn.cursor()
    success = False
    try:
        cursor.execute(f"DELETE FROM ratings WHERE userid = ? AND movieid = ? AND date = ?", (rating.userid, rating.movieid, rating.date))
        conn.commit()
        success = True
    except sqlite3.OperationalError:
        print("Operational issue")
    except sqlite3.DatabaseError:
        print("Database error")
    conn.close()
    return {"Success": success}


@api.delete("/delete_ratings", tags = ["delete_ratings"])
def delete_ratings(user: User):
    conn = connect_to_db("database.db")
    cursor = conn.cursor()
    success = False
    try:
        cursor.execute(f"DELETE FROM ratings WHERE userid = ?", (user.userid,))
        conn.commit()
        success = True
    except sqlite3.OperationalError:
        print("Operational issue")
    except sqlite3.DatabaseError:
        print("Database error")
    conn.close()
    return {"Success": success}

@api.patch("/update_rating/", tags = ["update_rating"])
def update_rating(update: Rating):
    """
    This is the update_rating route
    """
    conn = connect_to_db("database.db")
    cursor = conn.cursor()
    success = False
    try:
        cursor.execute(f"UPDATE users SET rating = ? WHERE userid = ? AND movieid = ? AND date = ?", (update.rating, update.userid, update.movieid, update.date))
        conn.commit()
        success = True
    except sqlite3.OperationalError:
        print("Operational issue")
    except sqlite3.DatabaseError:
        print("Database error")    
    return {"Success": success}
        

@api.post("/recommendation_system", tags = ["recommendation_system"])
async def recommendation_system(userid : int, movie : str):
    """
    This is the recommendation_system route.

    input : 
    
        userid : integer
        movie_title : string

    Return movies from a recommendation system
    """

    #recommendation_movies = hybrid_recommendation_movies(userid,movie)
    
    #return {f"When this route grows up it will provide recommendations for this movie: {movie}" : recommendation_movies}


@api.post("/log_event", tags = ["log_event"])
def log_event(event: Event):
    return {"This is the log_event route"}




