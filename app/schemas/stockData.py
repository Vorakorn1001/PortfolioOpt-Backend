from pydantic import BaseModel
from typing import Optional

class stockData(BaseModel):
    id: Optional[str]
    symbol: str
    name: str
    price: float
    annual5YrsReturn: float | None
    annual3YrsReturn: float | None
    annual1YrReturn: float
    ytdReturn: float
    volatility: float
    momentum: float
    beta: float
    sector: str
    industry: str
    marketCap: float | str
    dataCollectedDays: int

class stockDB(BaseModel):
    symbol: str
    name: str
    price: float
    annual5YrsReturn: float
    annual3YrsReturn: float
    annual1YrReturn: float
    fiveYrsVolatility: float
    threeYrsVolatility: float
    oneYrVolatility: float
    threeMthsMomentum: float
    ytdReturn: float
    sector: str
    industry: str
    marketCap: float | str
    dataCollectedDays: int