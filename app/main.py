from app.routes import stock, portfolio, optimize, ibkr, user
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi import FastAPI
import os

load_dotenv()

app = FastAPI()

origins = [
    "http://192.168.1.52:3000",
    "http://10.31.68.46:3000",
    os.getenv("FRONTEND_URL"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins
    allow_credentials=True,
    allow_methods=["*"],    # Allow all HTTP methods
    allow_headers=["*"],    # Allow all headers
)

# Include routes
app.include_router(stock.router, prefix="/stock")
app.include_router(portfolio.router, prefix="/portfolio")
app.include_router(optimize.router, prefix="/optimize")
app.include_router(ibkr.router, prefix="/ibkr")
app.include_router(user.router, prefix="/user")

@app.get("/")
def read_root():
    return {"message": "Nothing to see here, move along"}
