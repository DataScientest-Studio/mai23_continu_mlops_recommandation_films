from pydantic import BaseModel, EmailStr, SecretStr, constr, conint
from typing import Optional
import datetime as dt

# define User class
class User(BaseModel):
    """
    user's information from the database
    
    """
    userid: Optional[int] = None
    name: str
    email: EmailStr
    password: SecretStr
    _password_min_length = constr(min_length=6)

  
# define credentials class for authentication 
#class Credentials(BaseModel):
#    """
#    list of userid and password for authentication
#
#    """
#    userid: str
#    password: SecretStr
#    _password_min_length = constr(min_length=6)


# define class for logging activity
class Event(BaseModel):
    userid: int
    timestamp: float
    activity: str
    response_code: int
    response_message: str
    output: dict
    
# define class for ratings 
class Rating(BaseModel):
    """
    new_rating by userid from a movieid

    """
    userid: int
    movieid: int
    date: dt.datetime
    rating: int
    _rating_range = conint(ge=0,le=5)



    

    
    