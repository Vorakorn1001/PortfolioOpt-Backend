import sys
import os
from schemas.stockData import homePageStockData
from schemas.stockHistoryPrice import stockHistoryPrice
from models.database import db
from datetime import datetime
from typing import Dict, List
import asyncio
import pandas as pd
from yahoo_fin import stock_info as si
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

load_dotenv()

# Get the path to the project root dynamically
project_root = os.path.abspath(os.path.join(os.getcwd(), '../'))
sys.path.append(project_root)

class yahooService:
    def __init__(self, max_workers: int = 5):
        self.errorTickers = []
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
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

        def calculateReturn(stockHistoryPrice: pd.DataFrame, days: int) -> float:
            if len(stockHistoryPrice) >= days:
                start_price = stockHistoryPrice.iloc[-days]["close"]
                end_price = stockHistoryPrice.iloc[-1]["close"]
                return (end_price - start_price) / start_price
            return None

        def calculateYtdReturn(stockHistoryPrice: pd.DataFrame) -> float:
            current_year = datetime.now().year
            start_of_year = datetime(current_year, 1, 1)
            ytd_prices = stockHistoryPrice[stockHistoryPrice.index >= start_of_year]
            if len(ytd_prices) >= 2:
                start_price = ytd_prices.iloc[0]["close"]
                end_price = ytd_prices.iloc[-1]["close"]
                return (end_price - start_price) / start_price
            return None

        annual5YrsReturn = calculateReturn(stockHistoryPrice, numberOfDaysInYear * 5)
        annual3YrsReturn = calculateReturn(stockHistoryPrice, numberOfDaysInYear * 3)
        annual1YrReturn = calculateReturn(stockHistoryPrice, numberOfDaysInYear)
        ytdReturn = calculateYtdReturn(stockHistoryPrice)

        if annual5YrsReturn:
            annual5YrsReturn /= 5
        if annual3YrsReturn:
            annual3YrsReturn /= 3

        return {
            "symbol": stockData["Symbol"],
            "name": stockData["Name"],
            "price": stockHistoryPrice.iloc[-1]["close"],
            "annual5YrsReturn": annual5YrsReturn,
            "annual3YrsReturn": annual3YrsReturn,
            "annual1YrReturn": annual1YrReturn,
            "ytdReturn": ytdReturn,
            "sector": stockData["Sector"],
            "industry": stockData["Industry"],
            "marketCap": stockData["Market Cap"],
        }

    def convertToStockHistory(self, data: Dict) -> Dict:
        return {
            "date": data.name,
            "open": data["open"],
            "high": data["high"],
            "low": data["low"],
            "close": data["close"],
            "volume": data["volume"],
            "symbol": data["ticker"],
        }

    def insertData(self, stockData: homePageStockData, stockHistory: List[stockHistoryPrice]):
        try:
            self.stockDataCollection.insert_one(stockData)
            self.stockHistoryPriceCollection.insert_many(stockHistory)
        except Exception as e:
            print(f"Error inserting data for {stockData['symbol']}: {e}")
        print(f"Data inserted for {stockData['symbol']}")
        
    async def getStockData(self, ticker: str, delay: int = 1) -> None:
        try:
            print(f"Fetching data for {ticker}")
            loop = asyncio.get_event_loop()
            # Fetch data in a thread
            historyPrice = await loop.run_in_executor(
                self.executor, 
                si.get_data, 
                ticker, 
            )
            print(f"Data fetched for {ticker}")
            await asyncio.sleep(delay)  # Add delay to prevent rate limiting
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            self.errorTickers.append((ticker, e))
            return
        
        stockData = self.readNasdaq()
        stockData = stockData[stockData["Symbol"] == ticker].iloc[0]
        stockData = self.convertToStockData(stockData, historyPrice)
        
        historyPrice = [
            self.convertToStockHistory(dp) for _, dp in historyPrice.iterrows()
        ]
        
        self.insertData(stockData, historyPrice)
        return
    
    async def initDatabase(self, batchSize: int = 20, delayBetweenBatches: int = 5, delayBetweenRequests: int = 1, isTest: bool = False):
        self.checkBox()
        
        nasdaq = self.readNasdaq()
        if nasdaq is None:
            return
        tickers = self.getTickers(nasdaq)
        
        if isTest:
            tickers = tickers[:100]
            
        self.setUpDatabase()
        
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

        os.makedirs("logs", exist_ok=True)
        with open("logs/errorTickers.txt", "w") as f:
            for ticker, error in self.errorTickers:
                f.write(f"{ticker}: {error}\n")
    
