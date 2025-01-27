from app.routes import stock, portfolio, optimize, ibkr, user
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

app = FastAPI()

origins = [
    "http://192.168.1.52:3000",
    "http://localhost:3000",
    "https://portfolio-opt.vercel.app",
    "https://portfolio-opt-git-main-vorakorns-projects.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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
