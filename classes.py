from pydantic import BaseModel
from typing import Optional

# define User class
class User(BaseModel):
    userid: Optional[int]
    name: str
    email: str
    password: str
  
# define credentials class for authentication 
class Credentials(BaseModel):
    userid: str
    password: str

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
    userid: int
    movieid: int
    rating: float

    

    
    