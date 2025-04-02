from typing import List, Dict, Optional
from pydantic import BaseModel
from app.schemas.stockData import stockData
from app.schemas.investorView import investorView, investorViewInput
from datetime import datetime

class portfolio(BaseModel):
    assets: List[stockData]
    investorViews: Optional[List[investorView]] = None
    
class portfolioInput(BaseModel):
    assets: List[str]
    investorViews: Optional[List[investorViewInput]] = None

class portfolioData(BaseModel):
    activePortfolio: str
    portfolios: Dict[str, portfolioInput]
    
class portfolioDB(BaseModel):
    email: str
    name: str
    portfolio: portfolioInput
    createdAt: datetime
    updatedAt: datetime