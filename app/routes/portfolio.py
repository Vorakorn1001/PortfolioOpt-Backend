from pymongo import MongoClient
from fastapi import APIRouter, Depends, HTTPException
from app.models.database import db
import os
from pydantic import BaseModel
from typing import List
import pandas as pd
from app.dependency import getPortfolioService, getOptimizeService
from app.services.portfolioService import portfolioService
from app.services.optimizeService import optimizeService
from fastapi.responses import JSONResponse
import numpy as np


router = APIRouter()

class stockList(BaseModel):
    stocks: List[str]
    

class investorViewInput(BaseModel):
    asset1: str
    asset2: str | None
    percentage: float
    confidence: float

    
class investorView(BaseModel):
    P: List[List[int]] # Size of view x number of assets
    Q: List[float] # Size of view
    Omega: List[List[float]] # Size of view x size of view
    
    
def convertInvestorView(investorViews: List[investorViewInput], stocks: List[str]) -> investorView:
    P = np.zeros((len(investorViews), len(stocks)))
    Q = np.zeros(len(investorViews))
    Omega = np.zeros((len(investorViews), len(investorViews)))
    
    for i, view in enumerate(investorViews):
        P[i][stocks.index(view.asset1)] = 1
        if view.asset2:
            P[i][stocks.index(view.asset2)] = -1
        Q[i] = view.percentage
        Omega[i][i] = 1 - view.confidence
    
    return investorView(P=P.tolist(), Q=Q.tolist(), Omega=Omega.tolist()) 

class constraint(BaseModel):
    isReturn: bool
    percentage: float
    
def processResponse(data, round_param=2):
    """
    Recursively processes elements in a structure (list, np.ndarray, dict).
    - Converts np.ndarray to list.
    - Rounds float elements to the specified precision.
    
    Args:
        data: The data structure to process (list, np.ndarray, or dict).
        round_param: The precision to round floats to.
    
    Returns:
        The processed structure.
    """
    if isinstance(data, np.ndarray):
        data = data.tolist()  # Convert ndarray to list
    
    if isinstance(data, list):
        return [processResponse(element, round_param) for element in data]
    
    elif isinstance(data, dict):
        return {key: processResponse(value, round_param) for key, value in data.items()}
    
    elif isinstance(data, float):
        return round(data, round_param)
    
    else:
        return data

@router.post("/pre")
def getPortfolioPre(
    stockList: stockList,
    portfolioService: portfolioService = Depends(getPortfolioService)
    ):
    # Input:
    #   Stock name list
    # Output:
    #   Correlation Matrix
    #   Prior Returns
    
    try:
        if (len(stockList.stocks) < 2):
            return JSONResponse(
            content={"status": "Error", "detail": "Please provide at least 2 stocks"},
            status_code=400
            )
        stocks = sorted(stockList.stocks)
        stockDataList = db['stockData'].find({'symbol': {'$in': stocks}}).to_list()
        
        if len(stockDataList) != len(stocks):
            return JSONResponse(
            content={"status": "Error", "detail": "Can't find stock in our database"},
            status_code=404
            )
        
        longestDays = checkLongestDay(stockDataList)
        
        data = db['stockHistoryPrice'].find({'symbol': {'$in': stocks}}).sort('date', -1).limit(len(stocks) * longestDays)
        
        df = pd.DataFrame(list(data))
        
        if len(df) != len(stocks) * longestDays:
            return JSONResponse(
            content={"status": "Error", "detail": "Can't find the history price in our database"},
            status_code=404
            )
        
        df = df[['date', 'symbol', 'close']]
        df = df.pivot(index='date', columns='symbol', values='close')
        correlationMatrix = df.corr().values
        dfReturn = df.pct_change(fill_method=None).dropna()
        marketCap = [stock["marketCap"] for stock in stockDataList]
        covMatrix = dfReturn.cov() * 252
        priorReturns = portfolioService.getPriorReturns(marketCap, covMatrix)

        response_data = {
            "status": "Success",
            "stocks": stocks,
            "correlationMatrix": correlationMatrix,
            "priorReturns": priorReturns
        }
        response_data = processResponse(response_data)
        return JSONResponse(content=response_data, status_code=200)
    
    except KeyError as ke:
        return JSONResponse(content={"error": f"KeyError: {ke}"}, status_code=400)
    except ValueError as ve:
        return JSONResponse(content={"error": f"ValueError: {ve}"}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    

    
@router.post("/post")
def getPortfolioPost(
    stockList: stockList,
    constraint: constraint,
    investorView: List[investorViewInput] | None = None,
    portfolioService: portfolioService = Depends(getPortfolioService),
    optimizeService: optimizeService = Depends(getOptimizeService),
    riskFreeRate = 0.02
    ):
    # Input:
    #   Stock list: List of stock symbols.
    #   Investor's Views [] | None:
    #       asset1: str 
    #       asset2: str | None
    #       action: str
    #       percentage: float
    #       confidence: float
    #   Constraint:
    #       isReturn: Boolean flag for return-based constraint.
    #       percentage: Percentage constraint for the portfolio.
    # Output:
    #   Portfolio Weight: Calculated weights for each stock.
    #   Portfolio Metrics:
    #   Portfolio Diversification: (not now)
    #   Mean-Variance Graph (not now)
    
    try:
        stocks = sorted(stockList.stocks)
        stockDataList = db['stockData'].find({'symbol': {'$in': stocks}}).to_list()
        
        if len(stockDataList) != len(stocks):
            return JSONResponse(
            content={"status": "Error", "detail": "Can't find stock in our database"},
            status_code=404
            )
        
        longestDays = checkLongestDay(stockDataList)
        
        data = db['stockHistoryPrice'].find({'symbol': {'$in': stocks}}).sort('date', -1).limit(len(stocks) * longestDays)
        
        df = pd.DataFrame(list(data))
        
        if len(df) != len(stocks) * longestDays:
            return JSONResponse(
            content={"status": "Error", "detail": "Can't find the history price in our database"},
            status_code=404
            )
        
        df = df[['date', 'symbol', 'close']]
        df = df.pivot(index='date', columns='symbol', values='close')
        correlationMatrix = df.corr().values
        dfReturn = df.pct_change(fill_method=None).dropna()
        marketCap = [stock["marketCap"] for stock in stockDataList]
        returns = dfReturn.mean() * 252
        covMatrix = np.cov(dfReturn, rowvar=False) * 252
        priorReturns = portfolioService.getPriorReturns(marketCap, covMatrix)
        
        if investorView:
            investorView = convertInvestorView(investorView, stocks)
            posteriorReturns, posteriorCovMatrix = portfolioService.getPosteriorVariables(investorView.P, investorView.Q, investorView.Omega, priorReturns, covMatrix)
            finalReturns = posteriorReturns
            finalCovMatrix = posteriorCovMatrix
        else:
            finalReturns = priorReturns
            finalCovMatrix = covMatrix
        
        if constraint.isReturn:
            weights = optimizeService.optimizeFixedReturn(constraint.percentage, finalReturns, finalCovMatrix)
        else:
            weights = optimizeService.optimizeFixedVariance(constraint.percentage, finalReturns, finalCovMatrix)
            
        expectedReturn = portfolioService.getPortfolioReturn(weights, returns)
        expectedVariance = portfolioService.getPortfolioVariance(weights, covMatrix)
        sharpRatio = portfolioService.getPortfolioSharpeRatio(weights, returns, covMatrix, riskFreeRate)
        VaR = portfolioService.getPortfolioVaR(weights, returns, covMatrix, riskFreeRate)
        EstimateShortfall = portfolioService.getPortfolioES(weights, returns, covMatrix, riskFreeRate)
        
        response_data = {
            "status": "Success",
            "stocks": stocks,
            "weights": weights,
            "metrics": [
            {
                "label": "Expected Return",
                "value": expectedReturn
            },
            {
                "label": "Expected Variance",
                "value": expectedVariance
            },
            {
                "label": "Sharpe Ratio",
                "value": sharpRatio
            },
            {
                "label": "Value at Risk (VaR)",
                "value": VaR
            },
            {
                "label": "Expected Shortfall (ES)",
                "value": EstimateShortfall
            },
            {
                "label": "Max Drawdown",
                "value": -1
            }
            ],
            "Days": longestDays
        }
        response_data = processResponse(response_data)
        return JSONResponse(content=response_data, status_code=200)
    
    except KeyError as ke:
        return JSONResponse(content={"error": f"KeyError: {ke}"}, status_code=400)
    except ValueError as ve:
        return JSONResponse(content={"error": f"ValueError: {ve}"}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

def getTimeByName(name):
    if name not in ['annual5YrsReturn', 'annual3YrsReturn', 'annual1YrReturn']:
        return None
    return int(name.split('annual')[1].split('YrsReturn')[0])

def checkLongestDay(stockDataList):
    times = ['annual5YrsReturn', 'annual3YrsReturn', 'annual1YrReturn']
    for time in times:
        for stockData in stockDataList:
            if stockData[time] is None:
                break
        else:
            return getTimeByName(time) * 252
    minDays = float('inf')
    for stockData in stockDataList:
        if stockData['dataCollectedDays'] < minDays:
            minDays = stockData['dataCollectedDays']
    return minDays
