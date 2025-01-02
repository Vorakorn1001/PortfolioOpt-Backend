from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get MongoDB URL and database name from environment variables
MONGO_URL = os.getenv("MONGO_URL")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

# Initialize MongoDB client and database
client = MongoClient(MONGO_URL)
db = client[MONGO_DB_NAME]