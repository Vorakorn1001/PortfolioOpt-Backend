from pydantic import BaseModel
from typing import List

class investorViewInput(BaseModel):
    asset1: str
    asset2: str | None = None
    percentage: float
    confidence: float
    
class investorView(BaseModel):
    P: List[List[int]] 
    Q: List[float]
    Omega: List[List[float]]
    size: int