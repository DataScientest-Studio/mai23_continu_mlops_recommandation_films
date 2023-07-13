from pydantic import BaseModel
from typing import Optional

# define User class
class User(BaseModel):
    account_id: Optional[int]
    name: str
    email: str
    password: str
    
# define class for logging
class Log_Entry(BaseModel):
    account_id: int
    timestamp: float
    activity: str
    response_code: int
    response_message: str
    output: dict
    


    

    
    