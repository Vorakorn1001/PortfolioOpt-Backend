from app.schemas.investorView import investorViewInput, investorView
from app.services.PortfolioService import PortfolioService
from app.dependency import getPortfolioService
from fastapi.responses import JSONResponse
from fastapi import APIRouter, Depends
from app.models.database import db
from typing import List
import pandas as pd
import numpy as np
import math

PADDING = 2

router = APIRouter()

@router.post("/view")
def viewExtension(
    stocks: List[str],
    investorViews: List[investorViewInput],
    portfolioService: PortfolioService = Depends(getPortfolioService)
):
    try:
        if len(stocks) < 2:
            return JSONResponse(
                content={"status": "Error", "detail": "Please provide at least 2 stocks"},
                status_code=400
            )

        stocks.sort()
        stockDataList = db['stockData'].find({'symbol': {'$in': stocks}}).to_list()

        if len(stockDataList) != len(stocks):
            return JSONResponse(
                content={"status": "Error", "detail": "Can't find stock in our database"},
                status_code=404
            )

        longestDays = checkLongestDays(stockDataList)

        data = db['stockHistoryPrice'].find({'symbol': {'$in': stocks}}).sort('date', -1).limit(len(stocks) * longestDays)
        df = pd.DataFrame(list(data))

        if len(df) != len(stocks) * longestDays:
            return JSONResponse(
                content={"status": "Error", "detail": "Can't find the history price in our database"},
                status_code=404
            )

        df = df[['date', 'symbol', 'close']]
        df = df.pivot(index='date', columns='symbol', values='close')
        dfReturn = df.pct_change(fill_method=None).dropna()
        marketCap = [stock["marketCap"] for stock in stockDataList]
        covMatrix = (dfReturn.cov() * 252).values

        investorViews = convertInvestorView(investorViews, stocks, max([a[i] for i, a in enumerate(covMatrix)]))
        priorReturns = portfolioService.getPriorReturns(marketCap, covMatrix)
        posteriorReturns = portfolioService.getPosteriorReturns(investorViews.P, investorViews.Q, investorViews.Omega, priorReturns, covMatrix) if investorViews.size else priorReturns

        responseData = {
            "status": "Success",
            "priorReturns": priorReturns,
            "posteriorReturns": posteriorReturns,
            "limits": {
                "maxReturn": math.floor(max(posteriorReturns) * 100 - PADDING),
                "minReturn": math.ceil(min(posteriorReturns) * 100 + PADDING),
                "maxVolatility": math.floor(max([np.sqrt(a[i]) for i, a in enumerate(covMatrix)]) * 100 - PADDING),
                "minVolatility": math.ceil(np.sqrt(1 / np.sum(np.linalg.pinv(covMatrix))) * 100 - PADDING),
            }
        }

        return JSONResponse(content=processResponse(responseData), status_code=200)

    except KeyError as ke:
        return JSONResponse(content={"error": f"KeyError: {ke}"}, status_code=400)
    except ValueError as ve:
        return JSONResponse(content={"error": f"ValueError: {ve}"}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.post("/")
def portfolioPage(
    stocks: List[str],
    investorViews: List[investorViewInput],
    portfolioService: PortfolioService = Depends(getPortfolioService)
):
    try:
        if len(stocks) < 2:
            return JSONResponse(
                content={"status": "Error", "detail": "Please provide at least 2 stocks"},
                status_code=400
            )

        stocks.sort()
        stockDataList = db['stockData'].find({'symbol': {'$in': stocks}}).to_list()

        if len(stockDataList) != len(stocks):
            return JSONResponse(
                content={"status": "Error", "detail": "Can't find stock in our database"},
                status_code=404
            )

        longestDays = checkLongestDays(stockDataList)
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
        covMatrix = (dfReturn.cov() * 252).values

        investorViews = convertInvestorView(investorViews, stocks, max([a[i] for i, a in enumerate(covMatrix)]))
        priorReturns = portfolioService.getPriorReturns(marketCap, covMatrix)
        posteriorReturns = portfolioService.getPosteriorReturns(investorViews.P, investorViews.Q, investorViews.Omega, priorReturns, covMatrix) if investorViews.size else priorReturns

        responseData = {
            "status": "Success",
            "stocks": stocks,
            "correlationMatrix": correlationMatrix,
            "priorReturns": priorReturns,
            "posteriorReturns": posteriorReturns,
            "limits": {
                "maxReturn": math.floor(max(posteriorReturns) * 100 - PADDING),
                "minReturn": math.ceil(min(posteriorReturns) * 100 + PADDING),
                "maxVolatility": math.floor(max([np.sqrt(a[i]) for i, a in enumerate(covMatrix)]) * 100 - PADDING),
                "minVolatility": math.ceil(np.sqrt(1 / np.sum(np.linalg.pinv(covMatrix))) * 100 - PADDING),
            }
        }

        return JSONResponse(content=processResponse(responseData), status_code=200)

    except KeyError as ke:
        return JSONResponse(content={"error": f"KeyError: {ke}"}, status_code=400)
    except ValueError as ve:
        return JSONResponse(content={"error": f"ValueError: {ve}"}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
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

def getTimeByName(name):
    if name not in ['annual5YrsReturn', 'annual3YrsReturn', 'annual1YrReturn']:
        return None
    return int(name.split('annual')[1].split('YrsReturn')[0])

def checkLongestDays(stockDataList):
    times = ['annual5YrsReturn', 'annual3YrsReturn', 'annual1YrReturn']
    for time in times:
        if all(stockData[time] is not None for stockData in stockDataList):
            return getTimeByName(time) * 252

    return min(stockData['dataCollectedDays'] for stockData in stockDataList)
