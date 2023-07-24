from pydantic import BaseModel, EmailStr, SecretStr, constr, conint
from typing import Optional

# define User class
class User(BaseModel):
    userid: Optional[int] = None
    name: str
    email: EmailStr
    password: SecretStr
    _password_min_length = constr(min_length=6)

  
# define credentials class for authentication 
class Credentials(BaseModel):
    userid: str
    password: SecretStr
    _password_min_length = constr(min_length=6)


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
    rating: conint(ge=1, le=5)

    

    
    