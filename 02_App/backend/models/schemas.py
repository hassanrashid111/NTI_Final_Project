"""
Pydantic API Request/Response Schemas
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Health Check
class HealthResponse(BaseModel):
    status: str = "ok"
    system: str = "FavraAI Retail Intelligence Platform"
    version: str = "1.0.0"
    model_loaded: bool
    model_path: Optional[str] = None
    device: str = "GPU"

# Single Predict Request
class SinglePredictionRequest(BaseModel):
    store_nbr: int = Field(1, description="Store Number (1-54)")
    item_nbr: int = Field(22345, description="Item SKU Number")
    date: str = Field("2017-08-16", description="Forecast Date (YYYY-MM-DD)")
    onpromotion: int = Field(0, description="1 if on promotion, else 0")

class SinglePredictionResponse(BaseModel):
    store_nbr: int
    item_nbr: int
    date: str
    forecast_sales: float
    safety_stock: float
    reorder_point: float
    target_stock_level: float
    recommended_order_qty: int
    alert_status: str

# Forecast Summary Response
class ForecastSummaryResponse(BaseModel):
    total_forecast_16d: float
    daily_peak_sales: float
    avg_daily_demand: float
    sku_count: int
    store_count: int

# Inventory Summary Response
class InventorySummaryResponse(BaseModel):
    total_skus: int
    critical_understock: int
    optimal_stock: int
    overstock: int
    total_reorder_qty: int
    service_level_target: float = 0.95
    lead_time_days: int = 7

# Store Metric Schema
class StoreMetric(BaseModel):
    store_nbr: int
    city: str
    state: str
    type: str
    cluster: int
    forecast_16d: float
    reorder_risk_skus: int

# Model Info Response
class ModelInfoResponse(BaseModel):
    algorithm: str
    model_file: str
    rmsle: float
    rmse: float
    mae: float
    mape: float
    r2_score: float
    latency_ms: float
    device: str
    feature_count: int
    total_train_rows: int
