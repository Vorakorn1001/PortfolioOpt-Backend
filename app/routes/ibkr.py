from fastapi import APIRouter, Depends, File, UploadFile
from app.utils.helper import processResponse
from fastapi.responses import JSONResponse
from fastapi import HTTPException
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
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV file")
    
    try :
        data = pd.read_csv(file.file)
        stockList = getStockList(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    if len(stockList) == 0: raise HTTPException(status_code=400, detail="No stocks found in the file")
    
    try:
        stockData = list(collection.find({"symbol": {"$in": stockList}}))
        response = processResponse(stockData)
        response = {
            "portfolio": response
        }
        return JSONResponse(content=response, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
