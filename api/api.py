import email_validator
import pandas as pd
import numpy as np
from fastapi import FastAPI
from classes import User, Event, Rating
from api_recommendation import hybrid_recommendation_movies
import sqlite3

# with open("model.pkl", "rb") as pickled:
#   model = pickle.load(pickled)


api = FastAPI(
    title="API project Recommnendation System, MLOps May 2023",
    description="This is a Recommendation System API",
    version="0.2.1",
    version_detail="addition of delete_user route",
    openapi_tags=[
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
         "description": "This is the log_event route"}, ]
)


def connect_to_db(db):
    connection = sqlite3.connect(db)
    connection.row_factory = sqlite3.Row
    return connection


@api.get('/', tags=["home"])  # default route
def get_home():
    """
    This is the home route
    """
    return {"Welcome to our API. This is a work in progress."}


@api.post("/new_user", tags=["new_user"])
def new_user(user: User):
    """
    This is the new_user route
    """

    try:
        email_validator.validate_email(user.email, check_deliverability=False)
    except email_validator.EmailNotValidError as e:
        raise HTTPException(status_code=400, detail="Invalid email format")


    conn = connect_to_db("database.db")
    cursor = conn.cursor()
    success = False
    try:
        cursor.execute(f"INSERT INTO users (name, email, password) VALUES (?,?,?)",
                       (user.name, user.email, user.password.get_secret_value()))
        new_user_id = cursor.lastrowid
        conn.commit()
        return {"userid": new_user_id}
    except sqlite3.IntegrityError:
        print("User already exists")
        return {"User already exists"}
    except sqlite3.ProgrammingError:
        print("SQL syntax error")
    except sqlite3.OperationalError:
        print("Operational issue")
    except sqlite3.DatabaseError:
        print("Database error")
    conn.close()
    return {None}


@api.delete("/delete_user", tags=["delete_user"])
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
    return {f"Success: {success}"}


from fastapi import HTTPException


@api.patch("/update_user/", tags=["update_user"])
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
        conn.close()
        return {f"Success: {success}"}
    elif field == "email":
        value = update.email
        try:
            cursor.execute(f"UPDATE users SET email = ? WHERE userid = ?", (value, update.userid))
            conn.commit()
            success = True
        except sqlite3.OperationalError:
            print("Operational issue")
        except sqlite3.DatabaseError:
            print("Database error")
        conn.close()
        return {f"Success: {success}"}
    elif field == "password":
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
        return {f"Success: {success}"}
    else:
        conn.close()
        raise HTTPException(status_code=400, detail=f"Invalid field: {field}")


@api.post("/new_rating", tags=["new_rating"])
def new_rating(rating: Rating):
    """
    This is the new_rating route

    """
    conn = connect_to_db("database.db")
    cursor = conn.cursor()
    try:
        cursor.execute(f"INSERT INTO ratings (userid, movieid, rating) VALUES (?,?,?)",
                       (rating.userid, rating.movieid, rating.rating))
        new_rating_id = cursor.lastrowid
        conn.commit()
        return {"ratingid": new_rating_id}
    except sqlite3.IntegrityError:
        print("Rating already exists for this film by this user.")
    except sqlite3.ProgrammingError:
        print("SQL syntax error")
    except sqlite3.OperationalError:
        print("Operational issue")
    except sqlite3.DatabaseError:
        print("Database error")
    conn.close()
    return {None}


@api.delete("/delete_ratings", tags=["delete_ratings"])
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
    return {f"Success: {success}"}


@api.patch("/update_rating/", tags=["update_rating"])
def update_rating(new_rating: Rating):
    """
    This is the update_rating route
    """
    conn = connect_to_db("database.db")
    cursor = conn.cursor()
    success = False
    try:
        cursor.execute(f"UPDATE users SET rating = ? WHERE userid = ? AND movieid = ?",
                       (new_rating.rating, new_rating.userid, new_rating.movieid))
        conn.commit()
        success = True
    except sqlite3.OperationalError:
        print("Operational issue")
    except sqlite3.DatabaseError:
        print("Database error")
    return {f"Success: {success}"}


@api.post("/recommendation_system", tags=["recommendation_system"])
async def recommendation_system(userId: int, movie: str):
    """
    This is the recommendation_system route.

    input :

        userid : integer
        movie_title : string

    Return movies from a recommendation system
    """

    recommendation_movies = hybrid_recommendation_movies(userId, movie)

    return {f"When this route grows up it will provide recommendations for this movie: {movie}": recommendation_movies}


@api.post("/log_event", tags=["log_event"])
def log_event(event: Event):
    return {"This is the log_event route"}




