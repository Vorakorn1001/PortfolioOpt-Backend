import sys
import os
from app.schemas.stockData import stockData, stockDB
from app.schemas.stockHistoryPrice import stockHistoryPrice
from app.models.database import db
from datetime import datetime
from typing import Dict, List
import asyncio
import pandas as pd
from yahoo_fin import stock_info as si
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer

from dotenv import load_dotenv

load_dotenv()

# Get the path to the project root dynamically
project_root = os.path.abspath(os.path.join(os.getcwd(), '../'))
sys.path.append(project_root)

class YahooService:
    def __init__(self, max_workers: int = 5):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.SPYadjclose = None
        
    def prepareNasdaq(self) -> None:
        nasdaqFile = "./data/nasdaq-stockscreener.csv"
        nasdaqCompanyFile = "./data/nasdaq-listed-symbols.csv"
        df = pd.read_csv(nasdaqFile)
        dfCompany = pd.read_csv(nasdaqCompanyFile)

        df = df.merge(dfCompany[['Symbol', 'Company Name']], left_on='Symbol', right_on='Symbol', how='left')
        df['Name'] = df['Company Name']
        df = df.drop(columns=['Company Name'])
        df.to_csv("./data/nasdaq.csv", index=False)
        
    def readNasdaq(self):
        try:
            nasdaq = pd.read_csv("data/nasdaq.csv")
            return nasdaq
        except Exception as e:
            print(f"Error reading nasdaq.csv: {e}")
            return None
    
    def getTickers(self, nasdaq: pd.DataFrame) -> List[str]:
        nasdaq.drop(["IPO Year", "Net Change", "% Change"], axis=1, inplace=True)
        nasdaq.dropna(inplace=True)
        nasdaq.reset_index(drop=True, inplace=True)
        
        return nasdaq["Symbol"].astype(str).tolist()
    
    def setUpDatabase(self) -> None:
        try:
            db.create_collection(
                "stockHistoryPrice",
                timeseries={
                    "timeField": "date",
                    "metaField": "symbol",
                    "granularity": "hours",
                }
            )
        except Exception as e:
            print(f"Collection already exists or error: {e}")

        try:
            db.create_collection("stockData")
        except Exception as e:
            print(f"Collection already exists or error: {e}")
            
        self.stockHistoryPriceCollection = db["stockHistoryPrice"]
        self.stockDataCollection = db["stockData"]
    
        
    def checkBox(self) -> None:
        # Have nasdaq.csv
        try:
            pd.read_csv("data/nasdaq.csv")
            print("✅ nasdaq.csv found")
        except:
            print("❌ nasdaq.csv not found")
            return
        
        # Check the connection to MongoDB
        try:
            # The command is cheap and does not require auth
            db.command("ping")
            print("✅ Connected to MongoDB")
        except Exception as e:
            print(f"❌ Failed to connect to MongoDB: {e}")
            return
        
        # Check the connection to Yahoo Finance
        try:
            si.get_data("AAPL")
            print("✅ Connected to Yahoo Finance")
        except Exception as e:
            print(f"❌ Failed to connect to Yahoo Finance: {e}")
            return
        
    def convertToStockData(self, stockData: Dict, stockHistoryPrice: pd.DataFrame) -> Dict:
        numberOfDaysInYear = 252
        
        def calculateAnnualizedReturn(stockHistoryPrice: pd.DataFrame, days: int) -> float:
            if len(stockHistoryPrice) >= days:
                recent_prices = stockHistoryPrice.iloc[-days:]
                daily_returns = recent_prices["adjclose"].pct_change(fill_method=None).dropna()  # Compute daily returns
                average_daily_return = daily_returns.mean()  # Mean of daily returns
                annualized_return = average_daily_return * numberOfDaysInYear  # Annualize
                return annualized_return
            return None
        
        def calculateVolatility(stockHistoryPrice: pd.DataFrame, days: int) -> float:
            if len(stockHistoryPrice) >= days:
                recent_prices = stockHistoryPrice.iloc[-days:]
                daily_returns = recent_prices["adjclose"].pct_change(fill_method=None).dropna()
                volatility = daily_returns.std() * (numberOfDaysInYear ** 0.5)  # Annualize
                return volatility
            return None
        
        def calculateMomentum(stockHistoryPrice: pd.DataFrame, days: int) -> float:
            if len(stockHistoryPrice) >= days:
                recent_prices = stockHistoryPrice.iloc[-days:]
                momentum = (recent_prices["adjclose"].iloc[-1] - recent_prices["adjclose"].iloc[0]) / recent_prices["adjclose"].iloc[0]
                return momentum
            return None
        
        def calculateBeta(stockHistoryPrice: pd.DataFrame, days: int) -> float:
            if len(stockHistoryPrice) >= days:
                recent_prices = stockHistoryPrice.iloc[-days:]
                spy_returns = self.SPYadjclose.pct_change(fill_method=None).dropna()
                stock_returns = recent_prices["adjclose"].pct_change(fill_method=None).dropna()
                beta = stock_returns.cov(spy_returns) / spy_returns.var()
                return beta
            return None

        def calculateYtdReturn(stockHistoryPrice: pd.DataFrame) -> float:
            current_year = datetime.now().year
            start_of_year = datetime(current_year, 1, 1)
            ytd_prices = stockHistoryPrice[stockHistoryPrice.index >= start_of_year]
            if len(ytd_prices) >= 2:
                ytd_return = (ytd_prices.iloc[-1]["adjclose"] - ytd_prices.iloc[0]["adjclose"]) / ytd_prices.iloc[0]["adjclose"]
                return ytd_return
            return None
        
        days = len(stockHistoryPrice)
        if days >= 5 * numberOfDaysInYear:
            days = 5 * numberOfDaysInYear
        elif days >= 3 * numberOfDaysInYear:
            days = 3 * numberOfDaysInYear
        elif days >= numberOfDaysInYear:
            days = numberOfDaysInYear

        annual5YrsReturn = calculateAnnualizedReturn(stockHistoryPrice, numberOfDaysInYear * 5)
        annual3YrsReturn = calculateAnnualizedReturn(stockHistoryPrice, numberOfDaysInYear * 3)
        annual1YrReturn = calculateAnnualizedReturn(stockHistoryPrice, numberOfDaysInYear)
        ytdReturn = calculateYtdReturn(stockHistoryPrice)
        fiveYrsVolatility = calculateVolatility(stockHistoryPrice, numberOfDaysInYear * 5)
        threeYrsVolatility = calculateVolatility(stockHistoryPrice, numberOfDaysInYear * 3)
        oneYrVolatility = calculateVolatility(stockHistoryPrice, numberOfDaysInYear)
        threeMthsMomentum = calculateMomentum(stockHistoryPrice, 63)
        beta = calculateBeta(stockHistoryPrice, days)
        Return = calculateAnnualizedReturn(stockHistoryPrice, days)
        Volatility = calculateVolatility(stockHistoryPrice, days)

        return {
            "symbol": stockData["Symbol"],
            "name": stockData["Name"],
            "price": stockHistoryPrice.iloc[-1]["close"],
            "annual5YrsReturn": annual5YrsReturn,
            "annual3YrsReturn": annual3YrsReturn,
            "annual1YrReturn": annual1YrReturn,
            "ytdReturn": ytdReturn,
            "sector": stockData["Sector"],
            "fiveYrsVolatility": fiveYrsVolatility,
            "threeYrsVolatility": threeYrsVolatility,
            "oneYrVolatility": oneYrVolatility,
            "industry": stockData["Industry"],
            "Return": Return,
            "Volatility": Volatility,
            "beta": beta,
            "marketCap": stockData["Market Cap"],
            "threeMthsMomentum": threeMthsMomentum,
            "dataCollectedDays": len(stockHistoryPrice),
        }

    def convertToStockHistory(self, data: Dict) -> Dict:
        return {
            "date": data.name,
            "open": data["open"],
            "high": data["high"],
            "low": data["low"],
            "close": data["close"],
            "volume": data["volume"],
            "adjclose": data["adjclose"],
            "symbol": data["ticker"],
        }

    def insertData(self, stockData: stockDB, stockHistory: List[stockHistoryPrice]):
        try:
            self.stockDataCollection.insert_one(stockData)
            self.stockHistoryPriceCollection.insert_many(stockHistory)
        except Exception as e:
            print(f"Error inserting data for {stockData['symbol']}: {e}")
        print(f"Data inserted for {stockData['symbol']}")
        
    async def getNasdaqIndex(self) -> None:
        # SPY
        try:
            print(f"Fetching data for SPY")
            loop = asyncio.get_event_loop()
            # Fetch data in a thread
            historyPrice = await loop.run_in_executor(
                self.executor, 
                si.get_data, 
                "SPY", 
            )
            
            self.SPYadjclose = historyPrice["adjclose"]
            
            self.stockHistoryPriceCollection.insert_many([
                self.convertToStockHistory(dp) for _, dp in historyPrice.iterrows()
            ])
            
        except Exception as e:
            print(f"Error fetching data for SPY: {e}")
            with open("data/error.log", "a") as f:
                f.write(f"SPY: {e}\n")
            return
        
    async def getStockData(self, ticker: str, delay: int = 1) -> None:
        attempt = 0
        while attempt < 3:
            try:
                print(f"Fetching data for {ticker}")
                loop = asyncio.get_event_loop()
                # Fetch data in a thread
                historyPrice = await loop.run_in_executor(
                    self.executor, 
                    si.get_data, 
                    ticker, 
                )
                if len(historyPrice) < 252:
                    print(f"Skip {ticker} due to insufficient data")
                    return
                print(f"Data fetched for {ticker}")
                await asyncio.sleep(delay)
                break
            except Exception as e:
                attempt += 1
                print(f"Attempt {attempt}: Error fetching data for {ticker}: {e}")
                with open("data/error.log", "a") as f:
                    f.write(f"{ticker} attempt {attempt}: {e}\n")
                if attempt >= 3:
                    return
        
        stockData = self.readNasdaq()
        stockData = stockData[stockData["Symbol"] == ticker].iloc[0]
        stockData = self.convertToStockData(stockData, historyPrice)
        
        historyPrice = [
            self.convertToStockHistory(dp) for _, dp in historyPrice.iterrows()
        ]
        
        self.insertData(stockData, historyPrice)
        return
    
    def addMinMax(self):
        standardizedField = ["ReturnZscore", "VolatilityLog", "marketCapLog", "betaZscore", "threeMthsMomentumZscore"]
        NormalizeField = ["returnMinMax", "volatileMinMax", "marketCapMinMax", "betaMinMax","threeMthsMomentumMinMax"]
        
        stocks = pd.DataFrame(self.stockDataCollection.find({}))
        stocks.set_index("_id", inplace=True)
        
        Zscaler = StandardScaler()
        stocks["marketCapLog"] = np.log1p(stocks["marketCap"])
        stocks["VolatilityLog"] = np.log1p(stocks["Volatility"])
        stocks["ReturnZscore"] = Zscaler.fit_transform(stocks[["Return"]])
        stocks["betaZscore"] = Zscaler.fit_transform(stocks[["beta"]])
        stocks["threeMthsMomentumZscore"] = Zscaler.fit_transform(stocks[["threeMthsMomentum"]])
        
        minMaxScaler = MinMaxScaler(feature_range=[-1, 1])
        stocks[NormalizeField] = minMaxScaler.fit_transform(stocks[standardizedField]) 
        
        stocks = stocks.drop(columns=standardizedField)
        
        stocks = stocks.where(pd.notna(stocks), None)
        
        records = stocks.to_dict(orient="records")
        for record in records:
            self.stockDataCollection.update_one(
                {"symbol": record["symbol"]},  # Filter by the unique key 'symbol'
                {"$set": record},              # Update the document with new data
                upsert=True                    # Insert if no matching document exists
            )
        
    async def initDatabase(
        self, 
        batchSize: int = 20, 
        delayBetweenBatches: int = 5, 
        delayBetweenRequests: int = 1, 
        isTest: bool = False):
        
        self.checkBox()
        nasdaq = self.readNasdaq()
        if nasdaq is None:
            return
        tickers = self.getTickers(nasdaq)
        if isTest:
            tickers = tickers[:100]
        self.setUpDatabase()
        
        await self.getNasdaqIndex()
                
        tasks = []
        results = []

        async def getStockDataAsync(ticker: str, delay: int):
            await self.getStockData(ticker, delay)

        async def processTickers():
            for i, ticker in enumerate(tickers):  # Adjust the range as needed
                task = asyncio.create_task(getStockDataAsync(ticker, delay=delayBetweenRequests))
                tasks.append(task)
                if len(tasks) == batchSize:
                    # Execute batch and wait for results
                    batchResults = await asyncio.gather(*tasks)
                    results.extend(batchResults)
                    tasks.clear()
                    print(f"Batch {i // batchSize + 1} completed. Waiting {delayBetweenBatches} seconds before next batch.")
                    await asyncio.sleep(delayBetweenBatches)

            # Process any remaining tasks
            if tasks:
                results.extend(await asyncio.gather(*tasks))

        await processTickers()
        
        self.addMinMax()
    
