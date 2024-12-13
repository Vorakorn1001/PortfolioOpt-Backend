from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def get_items():
    return [{"item_id": 1, "name": "Item A"}, {"item_id": 2, "name": "Item B"}]
