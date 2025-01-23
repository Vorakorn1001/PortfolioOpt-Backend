from app.schemas.investorView import investorViewInput, investorView
from app.schemas.stockData import stockData
from typing import List, Optional, Dict
import numpy as np
import math
import re

def processResponse(data, roundParam=2):
    # Check for NaN
    if isinstance(data, float) and math.isnan(data):
        return None
     
    # Round the value if it's a float
    if isinstance(data, float):
        return round(data, roundParam)
        
    # Handle numpy arrays by converting them to lists
    if isinstance(data, np.ndarray):
        data = data.tolist()

    # If data is a list, recursively process each element
    if isinstance(data, list):
        return [processResponse(element, roundParam) for element in data]
    
    if isinstance(data, stockData):
        return processResponse(data.model_dump())

    # If data is a dictionary, recursively process each key-value pair
    elif isinstance(data, dict):
        # If '_id' is present, convert it to string
        if "_id" in data:
            data["id"] = str(data["_id"])
            del data["_id"]

        # Iterate over each key-value pair in the dictionary
        for key, value in data.items():
            # Recursively process nested structures (e.g., lists or dictionaries)
            data[key] = processResponse(value, roundParam)
            
            # Special formatting for 'marketCap'
            if key == "marketCap":
                if value >= 1_000_000_000_000:
                    data[key] = f"{value / 1_000_000_000_000:.1f}t"
                elif value >= 1_000_000_000:
                    data[key] = f"{value / 1_000_000_000:.1f}b"
                elif value >= 1_000_000:
                    data[key] = f"{value / 1_000_000:.1f}m"
        return data
    # Return the value as is if it's neither a dict, list, nor numpy array
    return data

def convertInvestorView(investorViews: List[investorViewInput], stocks: List[str], maxVariance: float, tolerance: float = 1e-5) -> investorView:
    P = np.zeros((len(investorViews), len(stocks)))
    Q = np.zeros(len(investorViews))
    Omega = np.zeros((len(investorViews), len(investorViews)))

    for i, view in enumerate(investorViews):
        P[i][stocks.index(view.asset1)] = 1
        if view.asset2:
            P[i][stocks.index(view.asset2)] = -1
        Q[i] = view.percentage / 100
        Omega[i][i] = maxVariance * ((1 - (view.confidence / 100)) / view.confidence + tolerance)

    return investorView(P=P.tolist(), Q=Q.tolist(), Omega=Omega.tolist(), size=len(investorViews))

def getYearByName(name):
    if name not in ['annual5YrsReturn', 'annual3YrsReturn', 'annual1YrReturn']:
        return None
    return int(name.split('annual')[1].split('YrsReturn')[0])

def checkLongestDays(stockDataList):
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

def convertToGraphFormat(diversification):
    nodes = [{"name": "Portfolio 100%"}]
    nodes += [{"name": f"{sector} {percentage * 100:.2f}%"} for sector, percentage in diversification.items() if percentage * 100 >= 0.01]
    links = [{"source": 0, "target": i, "value": percentage * 100} for i, (_, percentage) in enumerate(diversification.items(), start=1) if percentage * 100 >= 0.01]
    result = {
        "nodes": nodes,
        "links": links
    }
    return result

def generateQuery(
    searchTerm: Optional[str], 
    sectors: Optional[List[str]] = None, 
    marketCaps: Optional[List[str]] = None, 
) -> Dict:
    marketCapRange = {
        "Mega": {"$gt": 200_000_000_000},
        "Large": {"$gte": 10_000_000_000, "$lte": 200_000_000_000},
        "Medium": {"$gte": 2_000_000_000, "$lte": 10_000_000_000},
        "Small": {"$gte": 300_000_000, "$lte": 2_000_000_000},
        "Micro": {"$gte": 50_000_000, "$lte": 300_000_000},
        "Nano": {"$lt": 50_000_000}
    }
    
    query = {"$and": []}

    # Add sectors filter if provided
    if sectors:
        query["$and"].append({"sector": {"$in": sectors}})

    # Add market cap filter if provided
    if marketCaps and marketCapRange:
        marketCapsQuery = [marketCapRange[marketCap] for marketCap in marketCaps if marketCap in marketCapRange]
        if marketCapsQuery:
            query["$and"].append({"$or": [{"marketCap": cap} for cap in marketCapsQuery]})

    # Add search term filter if provided
    if searchTerm:
        escapedTerm = re.escape(searchTerm)
        query["$and"].append({
            "$or": [
                {"name": {"$regex": escapedTerm, "$options": "i"}},
                {"symbol": {"$regex": escapedTerm, "$options": "i"}}
            ]
        })

    # Remove $and if it's empty
    if not query["$and"]:
        query.pop("$and")

    return query

def generateRadarWeights(values):
    keys = [
        "returnMinMax",
        "volatileMinMax",
        "marketCapMinMax",
        "betaMinMax",
        "threeMthsMomentumMinMax"
    ]
    return dict(zip(keys, values))

def generatePipeline(
    searchTerm: Optional[str], 
    sectors: Optional[List[str]] = None, 
    marketCaps: Optional[List[str]] = None, 
    radar: Optional[List[int]] = None,
    skip: Optional[int] = 0,
    size: Optional[int] = 20,
    ascending: Optional[bool] = False
) -> Dict:
    query = generateQuery(searchTerm, sectors, marketCaps)
    pipeline = []
    if query:
        pipeline.append({"$match": query})
    if radar:
        weights = generateRadarWeights(radar)
        pipeline.append({
        "$addFields": {
            "weightedSum": {
                "$add": [
                    {"$multiply": ["$returnMinMax", weights["returnMinMax"]]},
                    {"$multiply": ["$volatileMinMax", weights["volatileMinMax"]]},
                    {"$multiply": ["$marketCapMinMax", weights["marketCapMinMax"]]},
                    {"$multiply": ["$betaMinMax", weights["betaMinMax"]]},
                    {"$multiply": ["$threeMthsMomentumMinMax", weights["threeMthsMomentumMinMax"]]}
                ]
            }
        }
    })
    pipeline.append({"$sort": {"weightedSum": 1 if ascending else -1}})
    pipeline.append({"$skip": skip})
    pipeline.append({"$limit": size})
    return pipeline

def reshapeStockData(data: List[Dict]) -> List[stockData]:
    reshaped_list = []
    stock_data_fields = stockData.__annotations__.keys()
    
    for item in data:
        reshaped_data = {}
        reshaped_data["id"] = str(item["_id"])
        for field in stock_data_fields:
            if field == "id":
                continue
            reshaped_data[field] = item.get(field, getattr(stockData, field, None))
        reshaped_list.append(stockData(**reshaped_data))
    
    return reshaped_list
        