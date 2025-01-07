from pymongo import MongoClient
from fastapi import APIRouter
import numpy as np
from app.models.database import db
import os

router = APIRouter()

def processResponse(item):
    # Remove _id from the document
    if "_id" in item:
        item["_id"] = str(item["_id"])
    # item.pop("_id", None)  # Safely remove _id if it exists
    for key, value in item.items():
        if isinstance(value, float):
            item[key] = round(value, 2)
        if key == "marketCap":
            if value >= 1_000_000_000_000:
                item[key] = f"{value / 1_000_000_000_000:.1f}t"
            elif value >= 1_000_000_000:
                item[key] = f"{value / 1_000_000_000:.1f}b"
            elif value >= 1_000_000:
                item[key] = f"{value / 1_000_000:.1f}m"
    return item


@router.get("/{length}")
def get_stock(length: int, size: int = 20):
    collection = db["stockData"]
    top_stocks = collection.find().sort("marketCap", -1).limit(size + length).to_list()[-size:]
    top_stocks_list = [processResponse(item) for item in top_stocks]
    return top_stocks_list

@router.get("/")
def get_stock_default(size: int = 20):
    return get_stock(0, size)
