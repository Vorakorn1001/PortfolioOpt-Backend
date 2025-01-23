from pydantic import BaseModel
from datetime import datetime

class stockHistoryPrice(BaseModel):
    date: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    adjclose: float
    volume: int
    