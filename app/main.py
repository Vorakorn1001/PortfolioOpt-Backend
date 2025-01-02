from fastapi import FastAPI
from app.routes import stock, users, items
from app.models.database import db
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    "http://192.168.1.52:3000",  # Allow your frontend
    "http://localhost:3000",   
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins
    allow_credentials=True,
    allow_methods=["*"],    # Allow all HTTP methods
    allow_headers=["*"],    # Allow all headers
)

# Include routes
app.include_router(users.router, prefix="/users")
app.include_router(items.router, prefix="/items")
app.include_router(stock.router, prefix="/stock")

@app.get("/")
def read_root():
    return {"message": "Welcome to my FastAPI project!"}
