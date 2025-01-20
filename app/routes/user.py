from app.utils.helper import processResponse
from fastapi.responses import JSONResponse
from app.schemas.userData import userData, userDB
from app.schemas.stockData import stockData
from app.schemas.investorView import investorView, investorViewInput
from app.schemas.portfolioData import portfolio, portfolioData, portfolioDB, portfolioInput
from app.models.database import db
from fastapi import APIRouter
from typing import List, Dict, Optional
from datetime import datetime

router = APIRouter()

def convertPortfolioDBtoPortfolioData(userDB: userData, portfolioDB: List[portfolioDB]) -> portfolioData:
    activePortfolio = userDB.activePortfolio 
    portfolios = {portfolio.name: portfolio.portfolio for portfolio in portfolioDB}
    return portfolioData(activePortfolio=activePortfolio, portfolios=portfolios)
        
def convertPortfolioDataToPortfolioDB(user: userData, portfolio: portfolioData) -> List[portfolioDB]:
    results = []
    for name, portfolio in portfolio.portfolios.items():
        results.append(
            portfolioDB(
                email=user.email, 
                name=name, 
                portfolio=portfolio,
                createdAt=datetime.now(),
                updatedAt=datetime.now()
                ))
    return results

def createEmptyPortfolio() -> portfolioData:
    return portfolioData(
        activePortfolio="Portfolio", 
        portfolios={
            "Portfolio": portfolio(assets=[], investorViews=[])
        }
    )
    
def convertId(data: Dict) -> Dict:
    if "id" in data:
        data["_id"] = data["id"]
        del data["id"]
    return data

def mergePortfolioData(user: userData, portfolioDBList: List[portfolioDB], portfolioData: portfolioData) -> List[portfolioDB]:
    # Convert portfolioDBList to a dictionary for quick lookup by activePortfolio
    portfolioDataName = list(portfolioData.portfolios.keys())
    portfolioDataDB = convertPortfolioDataToPortfolioDB(user, portfolioData)
        
    for portfolio in portfolioDBList:
        if portfolio["name"] in portfolioDataName:
            index = portfolioDataName.index(portfolio["name"])
            # Update the portfolio by Adding asset/investorView together and removing duplicates
            asset1 = portfolio["portfolio"]["assets"]
            asset2 = portfolioData.portfolios[portfolio["name"]].assets
            asset2Symbol = [x.symbol for x in asset2]
            
            asset2 += [stockData(**x) for x in asset1 if x["symbol"] not in asset2Symbol]
            portfolioDataDB[index].portfolio.assets = asset2
            
            investorView1 = portfolio["portfolio"]["investorViews"]
            investorView2 = portfolioData.portfolios[portfolio["name"]].investorViews
            investorView2Symbol = [x.asset1 + (x.asset2 if x.asset2 else "none") for x in investorView2]
            
            investorView2 += [investorViewInput(**x) for x in investorView1 if x["asset1"] + (x["asset2"] if x["asset2"] else "none") not in investorView2Symbol]
            portfolioDataDB[index].portfolio.investorViews = investorView2
        else:
            # Add the portfolio to the list
            portfolioDataDB.append(portfolioDB(**portfolio))
                
    return portfolioDataDB

# Frontend Use for signIn/login
# Get user's portfolio data
@router.post("/signIn")
def signIn(user: userData, portfolio: portfolioData):
    # Input: userData, portfolioData
    # Output: portfolioData
    # Get a local portfolio data and return the portfolio data
    try:            
        userCollection = db["usersData"]
        portfolioCollection = db["portfolioData"]
        userRecord = userCollection.find_one({"email": user.email})
        if userRecord is None:
            # Create a new user and portfolio data
            newUser = userDB(
                email=user.email, 
                name=user.name,
                image=user.image,
                activePortfolio="Portfolio", 
                createdAt=datetime.now(), 
                updatedAt=datetime.now()
                )
            userCollection.insert_one(newUser.model_dump())
            portfolioDatabase = convertPortfolioDataToPortfolioDB(newUser, portfolio)
            output = convertPortfolioDBtoPortfolioData(newUser, portfolioDatabase).model_dump()
            portfolioDatabase = [pf.model_dump() for pf in portfolioDatabase]
            portfolioCollection.insert_many(portfolioDatabase)
            return JSONResponse(content=output, status_code=200)
        else:
            # User exist, get the portfolio data and merge with the local portfolio data
            portfolioRecord = list(portfolioCollection.find({"email": user.email})) 
            newPortfolio = mergePortfolioData(user, portfolioRecord, portfolio)    
            newPortfolioDict = [pf.model_dump() if type(pf) != dict else pf for pf in newPortfolio]            
            portfolioCollection.delete_many({"email": user.email})
            if len(newPortfolioDict): portfolioCollection.insert_many(newPortfolioDict)
            output = convertPortfolioDBtoPortfolioData(user, newPortfolio).model_dump()
            return JSONResponse(content=output, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=400)
    except KeyError as ke:
        return JSONResponse(content={"error": f"KeyError: {ke}"}, status_code=400)
    except TypeError as te:
        return JSONResponse(content={"error": f"TypeError: {te}"}, status_code=400)


# Frontend Use when there are update in portfolioData
# Update and Merge user's portfolio data
@router.post("/updateAssets")
def Update(user: userData, portfolio: List[stockData]):
    # Input: userData, portfolioData
    # Output: status_code
    # Get a local portfolio data, update the user's portfolio data and return the updated portfolio data

    try:
        userCollection = db["usersData"]
        portfolioCollection = db["portfolioData"]
        userRecord = userCollection.find_one({"email": user.email})
        if userRecord is None: return JSONResponse(content={"error": "User not found"}, status_code=404)
        
        activePortfolio = userRecord["activePortfolio"]    
    
        portfolioDatabase = portfolioCollection.find_one({"email": user.email, "name": activePortfolio})
        if portfolioDatabase is None: return JSONResponse(content={"error": "Portfolio not found"}, status_code=404)
        
        portfolioDatabase["portfolio"]["assets"] = [x.model_dump() for x in portfolio]
        portfolioDatabase["updatedAt"] = datetime.now()
        portfolioCollection.replace_one({"email": user.email, "name": activePortfolio}, portfolioDatabase)
        
        return JSONResponse(content={"status": "success"}, status_code=200)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=400)
    except KeyError as ke:
        return JSONResponse(content={"error": f"KeyError: {ke}"}, status_code=400)
    except TypeError as te:
        return JSONResponse(content={"error": f"TypeError: {te}"}, status_code=400)
    
@router.post("/updatePortfolio")
def Update(user: userData, portfolio: portfolioInput):
    # Input: userData, portfolioData
    # Output: status_code
    # Get a local portfolio data, update the user's portfolio data and return the updated portfolio data

    try:
        userCollection = db["usersData"]
        portfolioCollection = db["portfolioData"]
        userRecord = userCollection.find_one({"email": user.email})
        if userRecord is None: return JSONResponse(content={"error": "User not found"}, status_code=404)
        
        activePortfolio = userRecord["activePortfolio"]    
    
        portfolioDatabase = portfolioCollection.find_one({"email": user.email, "name": activePortfolio})
        if portfolioDatabase is None: return JSONResponse(content={"error": "Portfolio not found"}, status_code=404)
        
        portfolioDatabase["portfolio"] = portfolio.model_dump()
        portfolioDatabase["updatedAt"] = datetime.now()
        
        portfolioCollection.replace_one({"email": user.email, "name": activePortfolio}, portfolioDatabase)
        
        return JSONResponse(content={"status": "success"}, status_code=200)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=400)
    except KeyError as ke:
        return JSONResponse(content={"error": f"KeyError: {ke}"}, status_code=400)
    except TypeError as te:
        return JSONResponse(content={"error": f"TypeError: {te}"}, status_code=400)