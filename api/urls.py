from fastapi import APIRouter
from api.views import router as pdf_router

api_router = APIRouter()
api_router.include_router(pdf_router, prefix="/api", tags=["PDF Processing"])
