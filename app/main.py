from fastapi import FastAPI
from app.routes import users, items

app = FastAPI()

# Include routes
app.include_router(users.router, prefix="/users", tags=["users"])
app.include_router(items.router, prefix="/items", tags=["items"])

@app.get("/")
def read_root():
    return {"message": "Welcome to my FastAPI project!"}
