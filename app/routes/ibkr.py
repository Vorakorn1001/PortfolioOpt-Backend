from fastapi import APIRouter, Depends, File, UploadFile
from app.utils.helper import processResponse
from fastapi.responses import JSONResponse
from app.models.database import db
import pandas as pd

collection = db["stockData"]

router = APIRouter()

def getStockList(df: pd.DataFrame) -> list:
    return [index[0] for index, row in df.iterrows() if index[-1] == "STK"]

@router.post("/")
def uploadFile(
    file: UploadFile = File(...)
):
    try:
        data = pd.read_csv(file.file)
        stockList = getStockList(data)
        stockData = list(collection.find({"symbol": {"$in": stockList}}))
        response = processResponse(stockData)
        response = {
            "portfolio": response
        }
        return JSONResponse(content=response, status_code=200)
    except Exception as e:
        return {"error": str(e)}
    except ValueError as ve:
        return {"error": str(ve)}
