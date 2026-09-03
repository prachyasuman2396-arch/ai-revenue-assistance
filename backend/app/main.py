from fastapi import FastAPI

from backend.app.api.routes import router as api_router
from backend.app.config.settings import settings

app = FastAPI(title=settings.app_name)

app.include_router(api_router, prefix="/api")


@app.get("/")
def root():
    return {"message": f"Welcome to {settings.app_name}"}
