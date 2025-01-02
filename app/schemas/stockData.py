from pydantic import BaseModel

class homePageStockData(BaseModel):
    symbol: str
    name: str
    price: float
    annual5YrsReturn: float
    annual3YrsReturn: float
    annual1YrReturn: float
    ytdReturn: float
    sector: str
    industry: str
    marketCap: float

class portfolioStockData(homePageStockData):
    impliedEqReturn: float
    