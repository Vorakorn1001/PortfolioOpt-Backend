from app.schemas.investorView import investorViewInput, investorView
from typing import List
import numpy as np

def processResponse(data, roundParam=2):
    # Round the value if it's a float
    if isinstance(data, float):
        return round(data, roundParam)
        
    # Handle numpy arrays by converting them to lists
    if isinstance(data, np.ndarray):
        data = data.tolist()

    # If data is a list, recursively process each element
    if isinstance(data, list):
        return [processResponse(element, roundParam) for element in data]

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