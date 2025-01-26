from pymongo import MongoClient
from fastapi import APIRouter, Depends, HTTPException
from app.models.database import db
import os
from pydantic import BaseModel
from typing import List
import pandas as pd
from app.dependency import getPortfolioService, getOptimizeService
from app.services.PortfolioService import PortfolioService
from app.services.OptimizeService import OptimizeService
from app.schemas.investorView import investorViewInput, investorView
from app.schemas.constraint import constraint
from fastapi.responses import JSONResponse
from app.utils.helper import processResponse, convertInvestorView, checkLongestDays, convertToGraphFormat
import numpy as np
from datetime import datetime

router = APIRouter()

PADDING = 2
DAYS = 1260

timeframeDict = {
    "ytd": int(np.busday_count(datetime(datetime.now().year, 1, 1).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))),
    "1m": 21,
    "3m": 63,
    "6m": 126,
    "1y": 252,
    '3y': 756,
    "5y": 1260
}

@router.post("/")
def optimize(
    stocks: List[str],
    constraint: constraint,
    investorViews: List[investorViewInput],
    timeframe: str,
    portfolioService: PortfolioService = Depends(getPortfolioService),
    optimizeService: OptimizeService = Depends(getOptimizeService),
    riskFreeRate=0.02,
    confidentLevel=0.95,
    volatilityStep=0.02
):
    try:
        stocks = sorted(stocks)
        stockDataList = db['stockData'].find({'symbol': {'$in': stocks}}).to_list()
        
        days = timeframeDict[timeframe]

        if len(stockDataList) != len(stocks):
            return JSONResponse(
                content={"status": "Error", "detail": "Can't find stock in our database"},
                status_code=404
            )

        longestDays = checkLongestDays(stockDataList)
        
        stockData = db['stockHistoryPrice'].find({'symbol': {'$in': stocks}}).sort('date', -1).limit(len(stocks) * longestDays)
        stockDf = pd.DataFrame(list(stockData))

        if len(stockDf) != len(stocks) * longestDays:
            return JSONResponse(
                content={"status": "Error", "detail": "Some history data is missing"},
                status_code=404
            )
            
        stockDf = stockDf[['date', 'symbol', 'close']].pivot(index='date', columns='symbol', values='close')

        marketData = db['stockHistoryPrice'].find({'symbol': "SPY"}).sort('date', -1).limit(longestDays)
        marketDf = pd.DataFrame(list(marketData))
        marketDf = marketDf[['date', 'symbol', 'close']].pivot(index='date', columns='symbol', values='close')
        
        dfReturn = stockDf.pct_change(fill_method=None).dropna()
        marketCap = [stock["marketCap"] for stock in stockDataList]
        returns = dfReturn.mean() * 252
        covMatrix = np.cov(dfReturn, rowvar=False) * 252
        priorReturns = portfolioService.getPriorReturns(marketCap, covMatrix)
        investorViews = convertInvestorView(investorViews, stocks, max([a[i] for i, a in enumerate(covMatrix)]))

        if investorViews.size > 0:
            posteriorReturns, posteriorCovMatrix = portfolioService.getPosteriorVariables(investorViews.P, investorViews.Q, investorViews.Omega, priorReturns, covMatrix)
            finalReturns = posteriorReturns
            finalCovMatrix = posteriorCovMatrix
        else:
            finalReturns = priorReturns
            finalCovMatrix = covMatrix
                        
        if constraint.isReturn:
            weights = optimizeService.optimizeFixedReturn(constraint.percentage, finalReturns, finalCovMatrix)
        else:
            weights = optimizeService.optimizeFixedRisk(constraint.percentage, finalReturns, finalCovMatrix)

        sectorWeights = portfolioService.getPortfolioSectorWeights(weights, stocks, stockDataList)
        sectorWeights = convertToGraphFormat(sectorWeights)

        portfolioSeries = stockDf.dot(weights).dropna()
        
        combinedDf = pd.concat([portfolioSeries, marketDf], axis=1).tail(min(days, longestDays))
                                
        combinedDf = combinedDf.rename(columns={'SPY': 'market', 0: 'portfolio'})
        combinedDf['portfolioReturn'] = combinedDf['portfolio'].pct_change(fill_method=None)
        combinedDf['marketReturn'] = combinedDf['market'].pct_change(fill_method=None)
        
        combinedDf[['marketReturn', 'portfolioReturn']] = combinedDf[['marketReturn', 'portfolioReturn']].fillna(0)
        
        cumulativeReturns = combinedDf[['portfolioReturn', 'marketReturn']].cumsum().dropna()
        
        metrics = portfolioService.getPortfolioMetrics(combinedDf, cumulativeReturns, confidentLevel, riskFreeRate)

        portfolioVsMarket = {
            "days": cumulativeReturns.index.strftime("%Y-%m-%d").tolist(),
            "portfolio": cumulativeReturns["portfolioReturn"].tolist(),
            "market": cumulativeReturns["marketReturn"].tolist(),
        }

        minVolatile = round(np.sqrt(1 / np.sum(np.linalg.pinv(covMatrix))), 3) + PADDING / 100
        maxVolatile = round(max([np.sqrt(covMatrix[x][x]) for x in range(len(covMatrix))]), 3) - PADDING / 100

        meanVarianceGraph = optimizeService.optimizeRangeRisk(minVolatile, maxVolatile, volatilityStep, returns, covMatrix, riskFreeRate)
        responseData = {
            "status": "Success",
            "stocks": stocks,
            "weights": weights,
            "metrics": metrics,
            "diversification": sectorWeights,
            "meanVarianceGraph": meanVarianceGraph,
            "portfolioVsMarket": portfolioVsMarket,
            "days": longestDays
        }
        responseData = processResponse(responseData, 4)
        return JSONResponse(content=responseData, status_code=200)

    except KeyError as ke:
        return JSONResponse(content={"error": f"KeyError: {ke}"}, status_code=400)
    except ValueError as ve:
        return JSONResponse(content={"error": f"ValueError: {ve}"}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.post("/change")
def change(
    stocks: List[str],
    weights: List[float],
    timeframe: str,
    portfolioService: PortfolioService = Depends(getPortfolioService),
    optimizeService: OptimizeService = Depends(getOptimizeService),
    riskFreeRate=0.02,
    confidentLevel=0.95,
    volatilityStep=0.01
):
    try:
        stocks = sorted(stocks)
        stockDataList = db['stockData'].find({'symbol': {'$in': stocks}}).to_list()
        
        days = timeframeDict[timeframe]

        if len(stockDataList) != len(stocks):
            return JSONResponse(
                content={"status": "Error", "detail": "Can't find stock in our database"},
                status_code=404
            )

        longestDays = checkLongestDays(stockDataList)

        stockData = db['stockHistoryPrice'].find({'symbol': {'$in': stocks}}).sort('date', -1).limit(len(stocks) * longestDays)
        stockDf = pd.DataFrame(list(stockData))

        if len(stockDf) != len(stocks) * longestDays:
            return JSONResponse(
                content={"status": "Error", "detail": "Some history data is missing"},
                status_code=404
            )

        stockDf = stockDf[['date', 'symbol', 'close']].pivot(index='date', columns='symbol', values='close')

        marketData = db['stockHistoryPrice'].find({'symbol': "SPY"}).sort('date', -1).limit(longestDays)
        marketDf = pd.DataFrame(list(marketData))
        marketDf = marketDf[['date', 'symbol', 'close']].pivot(index='date', columns='symbol', values='close')

        dfReturn = stockDf.pct_change(fill_method=None).dropna()
        returns = dfReturn.mean() * 252
        covMatrix = np.cov(dfReturn, rowvar=False) * 252

        sectorWeights = portfolioService.getPortfolioSectorWeights(weights, stocks, stockDataList)
        sectorWeights = convertToGraphFormat(sectorWeights)

        portfolioSeries = stockDf.dot(weights)
        combinedDf = pd.concat([portfolioSeries, marketDf], axis=1).tail(min(days, longestDays))
        combinedDf = combinedDf.rename(columns={'SPY': 'market', 0: 'portfolio'})
        
        combinedDf['portfolioReturn'] = combinedDf['portfolio'].pct_change(fill_method=None)
        combinedDf['marketReturn'] = combinedDf['market'].pct_change(fill_method=None)
        
        combinedDf[['marketReturn', 'portfolioReturn']] = combinedDf[['marketReturn', 'portfolioReturn']].fillna(0)
        
        cumulativeReturns = combinedDf[['portfolioReturn', 'marketReturn']].cumsum().dropna()
        
        metrics = portfolioService.getPortfolioMetrics(combinedDf, cumulativeReturns, confidentLevel, riskFreeRate)
        
        
        portfolioVsMarket = {
            "days": cumulativeReturns.index.strftime("%Y-%m-%d").tolist(),
            "portfolio": cumulativeReturns["portfolioReturn"].tolist(),
            "market": cumulativeReturns["marketReturn"].tolist(),
        }

        responseData = {
            "status": "Success",
            "stocks": stocks,
            "weights": weights,
            "metrics": metrics,
            "diversification": sectorWeights,
            "portfolioVsMarket": portfolioVsMarket,
            "days": longestDays
        }
        responseData = processResponse(responseData, 4)
        return JSONResponse(content=responseData, status_code=200)

    except KeyError as ke:
        return JSONResponse(content={"error": f"KeyError: {ke}"}, status_code=400)
    except ValueError as ve:
        return JSONResponse(content={"error": f"ValueError: {ve}"}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/performance")
def performance(
    stocks: List[str],
    weights: List[float],
    timeframe: str,
    portfolioService: PortfolioService = Depends(getPortfolioService),
    optimizeService: OptimizeService = Depends(getOptimizeService),
    riskFreeRate=0.02,
    confidentLevel=0.95,
    volatilityStep=0.01
):
    try:
        stocks = sorted(stocks)
        stockDataList = db['stockData'].find({'symbol': {'$in': stocks}}).to_list()

        days = timeframeDict[timeframe]
        
        if len(stockDataList) != len(stocks):
            return JSONResponse(
                content={"status": "Error", "detail": "Can't find stock in our database"},
                status_code=404
            )

        longestDays = checkLongestDays(stockDataList)

        stockData = db['stockHistoryPrice'].find({'symbol': {'$in': stocks}}).sort('date', -1).limit(len(stocks) * longestDays)
        stockDf = pd.DataFrame(list(stockData))

        if len(stockDf) != len(stocks) * longestDays:
            return JSONResponse(
                content={"status": "Error", "detail": "Some history data is missing"},
                status_code=404
            )

        stockDf = stockDf[['date', 'symbol', 'close']].pivot(index='date', columns='symbol', values='close')

        marketData = db['stockHistoryPrice'].find({'symbol': "SPY"}).sort('date', -1).limit(longestDays)
        marketDf = pd.DataFrame(list(marketData))
        marketDf = marketDf[['date', 'symbol', 'close']].pivot(index='date', columns='symbol', values='close')

        dfReturn = stockDf.pct_change(fill_method=None).dropna()
        returns = dfReturn.mean() * 252
        covMatrix = np.cov(dfReturn, rowvar=False) * 252

        portfolioSeries = stockDf.dot(weights)
        combinedDf = pd.concat([portfolioSeries, marketDf], axis=1).tail(min(days, longestDays))
        combinedDf = combinedDf.rename(columns={'SPY': 'market', 0: 'portfolio'})
        combinedDf['portfolioReturn'] = combinedDf['portfolio'].pct_change(fill_method=None)
        combinedDf['marketReturn'] = combinedDf['market'].pct_change(fill_method=None)
        
        combinedDf[['marketReturn', 'portfolioReturn']] = combinedDf[['marketReturn', 'portfolioReturn']].fillna(0)
        
        cumulativeReturns = combinedDf[['portfolioReturn', 'marketReturn']].cumsum().dropna()
        
        metrics = portfolioService.getPortfolioMetrics(combinedDf, cumulativeReturns, confidentLevel, riskFreeRate)
        
        
        portfolioVsMarket = {
            "days": cumulativeReturns.index.strftime("%Y-%m-%d").tolist(),
            "portfolio": cumulativeReturns["portfolioReturn"].tolist(),
            "market": cumulativeReturns["marketReturn"].tolist(),
        }

        responseData = {
            "status": "Success",
            "stocks": stocks,
            "weights": weights,
            "metrics": metrics,
            "portfolioVsMarket": portfolioVsMarket,
            "days": longestDays
        }
        responseData = processResponse(responseData, 4)
        return JSONResponse(content=responseData, status_code=200)
    
    except KeyError as ke:
        raise HTTPException(status_code=400, detail=f"KeyError: {ke}")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"ValueError: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


