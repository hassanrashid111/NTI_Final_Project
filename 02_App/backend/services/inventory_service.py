"""
Operations Research & Inventory Optimization Service
"""
import numpy as np

def calculate_inventory_metrics(
    daily_avg_demand: float,
    demand_std: float,
    current_stock: float,
    service_level: float = 0.95,
    lead_time_days: int = 7,
    review_period_days: int = 7,
):
    # Service factor Z mapping
    z_map = {0.90: 1.28, 0.95: 1.65, 0.98: 2.05, 0.99: 2.33}
    z_factor = z_map.get(service_level, 1.65)

    # 1. Safety Stock (SS)
    safety_stock = float(np.ceil(z_factor * max(0.1, demand_std) * np.sqrt(lead_time_days)))

    # 2. Reorder Point (ROP)
    reorder_point = float(np.ceil((daily_avg_demand * lead_time_days) + safety_stock))

    # 3. Target Stock Level (TSL)
    target_stock_level = float(np.ceil(reorder_point + (daily_avg_demand * review_period_days)))

    # 4. Recommended Order Quantity (ROQ)
    recommended_order_qty = int(max(0, target_stock_level - current_stock))

    # 5. Alert Status Classification
    if current_stock < reorder_point:
        alert_status = "CRITICAL_UNDERSTOCK"
    elif current_stock > target_stock_level:
        alert_status = "OVERSTOCK"
    else:
        alert_status = "OPTIMAL_STOCK"

    return {
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "target_stock_level": target_stock_level,
        "recommended_order_qty": recommended_order_qty,
        "alert_status": alert_status,
    }
