from pymongo import MongoClient
from fastapi import APIRouter
from app.models.database import db
import os

router = APIRouter()

def dropId(item):
    # Remove _id from the document
    item.pop("_id", None)  # Safely remove _id if it exists
    return item

@router.get("/")
def get_stock():
    collection = db["stock_info"]

    top_stocks = collection.find().sort("market_cap", -1).limit(20)
    top_stocks_list = [dropId(item) for item in top_stocks]

    return top_stocks_list
