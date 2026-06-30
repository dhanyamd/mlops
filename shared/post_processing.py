import os
import yaml
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve
from shared.observability.logging import get_logger

log = get_logger(__name__)


class ProbabilityCalibrator:
    """Calibrate prediction probabilities to represent true likelihoods."""

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.calibrator = None

    def fit(self, base_estimator, X_val: pd.DataFrame, y_val: pd.Series):
        """Fit calibrator on top of a pre-trained base estimator."""
        log.info("fitting_probability_calibrator", method=self.method, rows=len(X_val))
        self.calibrator = CalibratedClassifierCV(
            estimator=base_estimator,
            method=self.method,
            cv="prefit"
        )
        self.calibrator.fit(X_val, y_val)
        return self

    def calibrate(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated probabilities for the positive class."""
        if self.calibrator is None:
            raise ValueError("Calibrator is not fitted.")
        return self.calibrator.predict_proba(X)[:, 1]


class ThresholdOptimizer:
    """Find the optimal decision threshold based on business metrics or F-beta."""

    @staticmethod
    def optimize_fbeta(y_true: np.ndarray, y_proba: np.ndarray, beta: float = 1.0) -> float:
        """Find the threshold that maximizes the F-beta score."""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        
        # Avoid division by zero
        denominator = (beta**2 * precisions) + recalls
        f_scores = np.where(
            denominator == 0,
            0.0,
            (1 + beta**2) * (precisions * recalls) / denominator
        )
        
        best_idx = np.argmax(f_scores)
        best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
        log.info("optimized_fbeta_threshold", beta=beta, best_threshold=best_threshold, best_f_score=f_scores[best_idx])
        return best_threshold

    @staticmethod
    def optimize_business_cost(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        cost_fn: float = 15.0,  # Cost of False Negative (uncaught fraud)
        cost_fp: float = 1.0,   # Cost of False Positive (false alarm/declined transaction)
    ) -> float:
        """Find the threshold that minimizes the overall business cost.
        
        Total Cost = Cost_FN * False_Negatives + Cost_FP * False_Positives
        """
        thresholds = np.linspace(0.01, 0.99, 100)
        costs = []
        
        for t in thresholds:
            preds = (y_proba >= t).astype(int)
            fn = np.sum((preds == 0) & (y_true == 1))
            fp = np.sum((preds == 1) & (y_true == 0))
            total_cost = (cost_fn * fn) + (cost_fp * fp)
            costs.append(total_cost)
            
        best_idx = np.argmin(costs)
        best_threshold = float(thresholds[best_idx])
        log.info(
            "optimized_business_cost_threshold",
            best_threshold=best_threshold,
            min_cost=costs[best_idx],
            cost_fn=cost_fn,
            cost_fp=cost_fp,
        )
        return best_threshold


class BusinessRuleEngine:
    """Applies rule-based overrides on top of raw model predictions."""

    def __init__(self, rules_path: str | None = None):
        self.rules = {}
        if rules_path and os.path.exists(rules_path):
            with open(rules_path, "r") as f:
                self.rules = yaml.safe_load(f) or {}
            log.info("loaded_business_rules", path=rules_path, rules_count=len(self.rules))
        else:
            # Default fallback hardcoded rules
            self.rules = {
                "max_amount_force_block": 10000.0,
                "min_amount_force_approve": 5.0,
                "high_risk_score_threshold": 0.85,
            }

    def apply_rules(self, features: dict, model_score: float) -> dict:
        """Combine model score with hardcoded business policy rules.
        
        Returns:
            dict containing:
              - 'action': 'approve' | 'review' | 'block'
              - 'final_score': float
              - 'rule_triggered': str | None
        """
        amount = float(features.get("amount", 0.0))
        
        # Rule 1: High Transaction Value Limit
        max_limit = self.rules.get("max_amount_force_block", 10000.0)
        if amount >= max_limit and model_score > 0.3:
            return {
                "action": "block",
                "final_score": 1.0,
                "rule_triggered": "force_block_large_amount"
            }
            
        # Rule 2: Micro-transactions auto-approve
        min_limit = self.rules.get("min_amount_force_approve", 5.0)
        if amount <= min_limit and model_score < 0.7:
            return {
                "action": "approve",
                "final_score": model_score,
                "rule_triggered": "force_approve_micro_transaction"
            }
            
        # Standard threshold checks
        high_threshold = self.rules.get("high_risk_score_threshold", 0.85)
        
        if model_score >= high_threshold:
            return {
                "action": "block",
                "final_score": model_score,
                "rule_triggered": "high_model_score"
            }
        elif model_score >= 0.5:
            return {
                "action": "review",
                "final_score": model_score,
                "rule_triggered": "medium_model_score"
            }
        else:
            return {
                "action": "approve",
                "final_score": model_score,
                "rule_triggered": None
            }
