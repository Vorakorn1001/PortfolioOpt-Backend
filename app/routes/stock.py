from app.utils.helper import processResponse, generatePipeline, reshapeStockData
from fastapi.responses import JSONResponse
from app.models.database import db
from fastapi import APIRouter, HTTPException
from app.schemas.filterData import filterData
from typing import List
import pandas as pd


router = APIRouter()

@router.get("/")
def getStock(
    skip: int, 
    size: int = 20
    ):
    try:
        collection = db["stockData"]
        top_stocks = (
            collection.find()
            .sort("marketCap", -1)
            .skip(skip)
            .limit(size)
        )
        top_stocks_list = processResponse(list(top_stocks))
        return JSONResponse(content=top_stocks_list, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/filter")
def getStockFilter(
    sectors: List[str],
    marketCaps: List[str],
    radar: List[int],
    keyword: str = "",
    ascending: bool = False,
    skip: int = 0,
    size: int = 20
    ):
    if len(radar) != 5:
        raise HTTPException(status_code=400, detail="Radar data must have 5 values")
    try:
        radar = [int(value) - 3 for value in radar]
        if radar.count(0) == 5:
            radar[2] = 1
        radar = [value / 3 for value in radar]
        pipeline = generatePipeline(
            searchTerm=keyword, 
            sectors=sectors, 
            marketCaps=marketCaps, 
            radar=radar, 
            skip=skip, 
            size=size, 
            ascending=ascending)
        collection = db["stockData"]
        top_stocks = collection.aggregate(pipeline)
        top_stocks_list = reshapeStockData(list(top_stocks))
        top_stocks_list = processResponse(top_stocks_list)
        return JSONResponse(content=top_stocks_list, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))