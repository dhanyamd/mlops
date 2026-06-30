import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend to prevent GUI thread issues
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from shared.observability.logging import get_logger

log = get_logger(__name__)


class SHAPExplainer:
    """Wrapper around SHAP (SHapley Additive exPlanations) for model explainability."""

    def __init__(self, model, X_reference: pd.DataFrame):
        self.model = model
        try:
            self.explainer = shap.TreeExplainer(model)
        except Exception:
            # Sample reference to speed up generic explainers if it's too large
            background_data = (
                X_reference.sample(min(100, len(X_reference)), random_state=42)
                if len(X_reference) > 100
                else X_reference
            )
            self.explainer = shap.Explainer(model, background_data)

    def generate_summary_plot(self, X: pd.DataFrame, output_path: str) -> None:
        """Generate and save global SHAP summary plot."""
        log.info("generating_shap_summary_plot", rows=len(X))
        plt.figure(figsize=(10, 6))
        
        # Calculate shap values
        shap_values = self.explainer(X)
        shap.summary_plot(shap_values, X, show=False)
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close()
        log.info("saved_shap_summary_plot", path=output_path)

    def explain_prediction(self, X_instance: pd.DataFrame, top_n: int = 5) -> dict[str, float]:
        """Generate local explanations for a single prediction instance.
        
        Returns the top_n features contributing to the prediction.
        """
        shap_values = self.explainer(X_instance)
        # Extract shap values for the first row
        row_values = shap_values.values[0]
        # For classifier outputs with shape (num_features, num_classes) or similar
        if len(row_values.shape) > 1:
            row_values = row_values[:, 1]
            
        feature_names = X_instance.columns
        explanations = dict(zip(feature_names, row_values))
        
        # Sort by absolute SHAP value
        sorted_explanations = sorted(
            explanations.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return {name: float(val) for name, val in sorted_explanations[:top_n]}
