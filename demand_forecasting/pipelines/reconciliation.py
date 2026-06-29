"""Hierarchical Forecast Reconciliation module.

Why hierarchical reconciliation:
  In retail and supply chain, forecasts are generated at multiple levels:
  - Base level: individual products (e.g., S01_I01)
  - Top level: total store sales or total company sales
  
  If you forecast top-level and base-level independently, they will not sum
  consistently (i.e., sum(base_forecasts) != top_forecast).
  Hierarchical reconciliation uses linear algebra projections (Bottom-Up, OLS)
  to adjust all forecasts so they are mathematically consistent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shared.observability.logging import get_logger

log = get_logger(__name__)


class HierarchicalReconciler:
    """Reconciles forecasts across different aggregation levels using linear algebra."""

    def __init__(self, n_products: int):
        self.n_products = n_products
        # Summing matrix S: relates bottom-level to top-level.
        # For a simple hierarchy of Total (top) and N Products (bottom):
        # S has shape (n_products + 1, n_products)
        # S = [[1, 1, ..., 1],  <-- Total is sum of all products
        #      [1, 0, ..., 0],  <-- Product 1
        #      [0, 1, ..., 0],  <-- Product 2
        #      ...]
        self.S = np.vstack([
            np.ones((1, n_products)),  # Top level (Total)
            np.eye(n_products)        # Bottom level (Products)
        ])

    def reconcile_bottom_up(self, product_forecasts: np.ndarray) -> tuple[float, np.ndarray]:
        """Reconcile using the Bottom-Up approach.
        
        The top-level forecast is simply replaced by the sum of bottom-level forecasts.
        """
        # S @ bottom_forecasts
        reconciled = self.S @ product_forecasts
        total_forecast = float(reconciled[0])
        reconciled_products = reconciled[1:]
        return total_forecast, reconciled_products

    def reconcile_ols(self, top_forecast: float, product_forecasts: np.ndarray) -> tuple[float, np.ndarray]:
        """Reconcile using Ordinary Least Squares (OLS) MinTrace reconciliation.
        
        Adjusts all forecasts (top and bottom) to minimize the sum of squared differences,
        projecting them into the consistent space.
        
        Formula:
            y_reconciled = S @ (S^T @ S)^(-1) @ S^T @ y_unreconciled
        """
        y_unreconciled = np.concatenate([[top_forecast], product_forecasts])
        
        # OLS projection matrix: P = (S^T @ S)^(-1) @ S^T
        S_T = self.S.T
        try:
            inv_S_T_S = np.linalg.inv(S_T @ self.S)
            P = inv_S_T_S @ S_T
            
            # Reconciled bottom-level forecasts
            bottom_reconciled = P @ y_unreconciled
            # Project up to all levels (top + bottom)
            y_reconciled = self.S @ bottom_reconciled
            
            total_reconciled = float(y_reconciled[0])
            products_reconciled = y_reconciled[1:]
            
            # Ensure no negative forecasts are introduced by OLS projection
            products_reconciled = np.maximum(0, products_reconciled)
            total_reconciled = float(np.sum(products_reconciled))
            
            return total_reconciled, products_reconciled
        except np.linalg.LinAlgError as exc:
            log.warning("ols_reconciliation_failed_using_bu", error=str(exc))
            return self.reconcile_bottom_up(product_forecasts)
