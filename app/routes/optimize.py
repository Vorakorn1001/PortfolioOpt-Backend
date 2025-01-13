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
import numpy as np

router = APIRouter()

PADDING = 2

@router.post("/")
def optimize(
    stocks: List[str],
    constraint: constraint,
    investorViews: List[investorViewInput],
    days: int = 252,
    portfolioService: PortfolioService = Depends(getPortfolioService),
    optimizeService: OptimizeService = Depends(getOptimizeService),
    riskFreeRate=0.02,
    confidentLevel=0.95,
    volatilityStep=0.01
):
    try:
        stocks = sorted(stocks)
        stockDataList = db['stockData'].find({'symbol': {'$in': stocks}}).to_list()

        if len(stockDataList) != len(stocks):
            return JSONResponse(
                content={"status": "Error", "detail": "Can't find stock in our database"},
                status_code=404
            )

        longestDays = checkLongestDay(stockDataList)

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
            finalReturns = returns
            finalCovMatrix = covMatrix

        if constraint.isReturn:
            weights = optimizeService.optimizeFixedReturn(constraint.percentage, finalReturns, finalCovMatrix)
        else:
            weights = optimizeService.optimizeFixedRisk(constraint.percentage, finalReturns, finalCovMatrix)

        sectorWeights = portfolioService.getPortfolioSectorWeights(weights, stocks, stockDataList)
        sectorWeights = convertToGraphFormat(sectorWeights)

        portfolioSeries = stockDf.dot(weights)
        combinedDf = pd.concat([portfolioSeries, marketDf], axis=1).tail(min(days, longestDays))
        combinedDf = combinedDf.rename(columns={'SPY': 'market', 0: 'portfolio'})
        combinedDf['portfolioReturn'] = combinedDf['portfolio'].pct_change()

        metrics = portfolioService.getPortfolioMetrics(weights, returns, covMatrix, combinedDf['portfolioReturn'], confidentLevel, riskFreeRate)

        combinedDf['marketReturn'] = combinedDf['market'].pct_change()
        cumulativeReturns = combinedDf[['portfolioReturn', 'marketReturn']].cumsum().dropna()
        portfolioVsMarket = {
            "days": cumulativeReturns.index.strftime("%Y-%m-%d").tolist(),
            "portfolio": cumulativeReturns["portfolioReturn"].tolist(),
            "market": cumulativeReturns["marketReturn"].tolist(),
        }

        minVolatile = round(np.sqrt(1 / np.sum(np.linalg.pinv(covMatrix))), 2) + PADDING / 100
        maxVolatile = round(max([np.sqrt(covMatrix[x][x]) for x in range(len(covMatrix))]), 2) - PADDING / 100

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
    days: int = 252,
    portfolioService: PortfolioService = Depends(getPortfolioService),
    optimizeService: OptimizeService = Depends(getOptimizeService),
    riskFreeRate=0.02,
    confidentLevel=0.95,
    volatilityStep=0.01
):
    try:
        stocks = sorted(stocks)
        stockDataList = db['stockData'].find({'symbol': {'$in': stocks}}).to_list()

        if len(stockDataList) != len(stocks):
            return JSONResponse(
                content={"status": "Error", "detail": "Can't find stock in our database"},
                status_code=404
            )

        longestDays = checkLongestDay(stockDataList)

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
        combinedDf['portfolioReturn'] = combinedDf['portfolio'].pct_change()
        metrics = portfolioService.getPortfolioMetrics(weights, returns, covMatrix, combinedDf['portfolioReturn'], confidentLevel, riskFreeRate)
        combinedDf['marketReturn'] = combinedDf['market'].pct_change()
        cumulativeReturns = combinedDf[['portfolioReturn', 'marketReturn']].cumsum().dropna()
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


def convertToGraphFormat(diversification):
    nodes = [{"name": "Portfolio 100%"}]
    nodes += [{"name": f"{sector} {percentage * 100:.2f}%"} for sector, percentage in diversification.items() if percentage * 100 >= 0.01]
    links = [{"source": 0, "target": i, "value": percentage * 100} for i, (_, percentage) in enumerate(diversification.items(), start=1) if percentage * 100 >= 0.01]
    result = {
        "nodes": nodes,
        "links": links
    }
    return result

def convertInvestorView(investorViews: List[investorViewInput], stocks: List[str], maxVariance: float) -> investorView:
    P = np.zeros((len(investorViews), len(stocks)))
    Q = np.zeros(len(investorViews))
    Omega = np.zeros((len(investorViews), len(investorViews)))

    for i, view in enumerate(investorViews):
        P[i][stocks.index(view.asset1)] = 1
        if view.asset2:
            P[i][stocks.index(view.asset2)] = -1
        Q[i] = view.percentage / 100
        Omega[i][i] = maxVariance * (1 - (view.confidence / 100)) / view.confidence

    return investorView(P=P.tolist(), Q=Q.tolist(), Omega=Omega.tolist(), size=len(investorViews))

def processResponse(data, roundParam=2):
    if isinstance(data, np.ndarray):
        data = data.tolist()

    if isinstance(data, list):
        return [processResponse(element, roundParam) for element in data]

    elif isinstance(data, dict):
        return {key: processResponse(value, roundParam) for key, value in data.items()}

    elif isinstance(data, float):
        return round(data, roundParam)

    else:
        return data

def getYearByName(name):
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
            return getYearByName(time) * 252
    minDays = float('inf')
    for stockData in stockDataList:
        if stockData['dataCollectedDays'] < minDays:
            minDays = stockData['dataCollectedDays']
    return minDays
