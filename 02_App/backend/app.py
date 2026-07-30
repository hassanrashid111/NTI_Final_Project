"""
FavraAI — FastAPI Server Entrypoint
Predict. Optimize. Never Run Out.
"""
import os, sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add APP_DIR (02_App) and PROJECT_ROOT to sys.path so 'from backend ...' and 'import config' work seamlessly
BACKEND_DIR = Path(__file__).parent.resolve()
APP_DIR = BACKEND_DIR.parent.resolve()
PROJECT_ROOT = APP_DIR.parent.resolve()

for p in [str(APP_DIR), str(BACKEND_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from backend import config
    from backend.api.routes import router as api_router
    from backend.services.model_service import ModelService
except ModuleNotFoundError:
    import config
    from api.routes import router as api_router
    from services.model_service import ModelService

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: Load trained model ONCE into memory
    print("🚀 Initializing FavraAI Retail Intelligence Platform...")
    ms = ModelService.get_instance()
    print(f"✅ FavraAI Server Ready! Model loaded: {ms.model_path or 'Fallback Engine'}")
    yield
    # SHUTDOWN
    print("🛑 FavraAI Server Shutting Down...")

app = FastAPI(
    title="FavraAI — Retail Intelligence Platform API",
    description="Enterprise API for Demand Forecasting & Operations Research Inventory Control",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(config, 'CORS_ORIGINS', ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router, prefix=getattr(config, 'API_PREFIX', "/api/v1"))

# Serve sample datasets & Frontend SPA
SAMPLE_DIR = APP_DIR / "sample_data"
if SAMPLE_DIR.exists():
    app.mount("/sample_data", StaticFiles(directory=str(SAMPLE_DIR)), name="sample_data")

FRONTEND_DIR = APP_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app" if (APP_DIR / "backend").exists() else "app:app", host=getattr(config, 'HOST', '0.0.0.0'), port=getattr(config, 'PORT', 8000), reload=True)
