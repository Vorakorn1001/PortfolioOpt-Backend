from app.services.YahooService import YahooService
from app.services.PortfolioService import PortfolioService
from app.services.OptimizeService import OptimizeService

def getYahooService() -> YahooService:
    return YahooService()

def getPortfolioService() -> PortfolioService:
    return PortfolioService()

def getOptimizeService() -> OptimizeService:
    return OptimizeService()