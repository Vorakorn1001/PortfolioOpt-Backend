from pydantic import BaseModel

class constraint(BaseModel):
    isReturn: bool
    percentage: float