from pydantic import BaseModel
from typing import Optional

class stockData(BaseModel):
    id: Optional[str]
    symbol: str
    name: str
    price: float
    annual5YrsReturn: float
    annual3YrsReturn: float
    annual1YrReturn: float
    ytdReturn: float
    sector: str
    industry: str
    marketCap: float | str
    dataCollectedDays: int
    