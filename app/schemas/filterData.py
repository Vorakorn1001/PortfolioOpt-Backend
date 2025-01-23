from pydantic import BaseModel
from typing import List

class filterData(BaseModel):
    sectors: List[str] | List[None] = []
    marketCaps: List[str] | List[None] = []
    radar: List[int] | List[None] = []
    keyword: str | None = None
    