from pydantic import BaseModel
from datetime import datetime

class stockHistoryPrice(BaseModel):
    date: datetime
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    