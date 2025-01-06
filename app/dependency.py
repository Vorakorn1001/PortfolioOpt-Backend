from app.services.yahooService import yahooService
from app.services.portfolioService import portfolioService
from app.services.optimizeService import optimizeService

def getYahooService() -> yahooService:
    return yahooService()

def getPortfolioService() -> portfolioService:
    return portfolioService()

def getOptimizeService() -> optimizeService:
    return optimizeService()