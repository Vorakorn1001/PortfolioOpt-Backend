from app.utils.helper import processResponse
from fastapi.responses import JSONResponse
from app.models.database import db
from fastapi import APIRouter

router = APIRouter()

@router.get("/{length}")
def get_stock(length: int, size: int = 20):
    try:
        collection = db["stockData"]
        skip = length
        top_stocks = (
            collection.find()
            .sort("marketCap", -1)
            .skip(skip)
            .limit(size)
        )
        top_stocks_list = processResponse(list(top_stocks))
        return JSONResponse(content=top_stocks_list, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=400)
    except KeyError as ke:
        return JSONResponse(content={"error": f"KeyError: {ke}"}, status_code=400)

@router.get("/")
def get_stock_default(size: int = 20):
    return get_stock(0, size)
