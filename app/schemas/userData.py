from pydantic import BaseModel
from datetime import datetime

class userDB(BaseModel):
    name: str
    email: str
    image: str
    activePortfolio: str
    createdAt: datetime
    updatedAt: datetime
    
class userData(BaseModel):
    name: str
    email: str
    image: str
    activePortfolio: str