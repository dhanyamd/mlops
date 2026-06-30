import os
import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class AlertManager:
    """
    Handles sending alerts to external services like Slack and PagerDuty.
    """
    def __init__(self):
        self.slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
        self.pagerduty_routing_key = os.environ.get("PAGERDUTY_ROUTING_KEY")

    def send_slack_alert(self, message: str, level: str = "INFO"):
        """Sends an alert to Slack."""
        if not self.slack_webhook_url:
            logger.warning("SLACK_WEBHOOK_URL not set. Skipping Slack alert.")
            return

        emoji = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "🚨",
            "CRITICAL": "💥"
        }.get(level, "💬")

        payload = {
            "text": f"{emoji} *{level}* | MLOps Alert\n{message}"
        }

        try:
            response = requests.post(self.slack_webhook_url, json=payload, timeout=5)
            response.raise_for_status()
            logger.info(f"Slack alert sent successfully: {message}")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")

    def send_pagerduty_alert(self, summary: str, source: str, severity: str = "error", custom_details: Optional[dict] = None):
        """Sends an incident alert to PagerDuty."""
        if not self.pagerduty_routing_key:
            logger.warning("PAGERDUTY_ROUTING_KEY not set. Skipping PagerDuty alert.")
            return

        payload = {
            "routing_key": self.pagerduty_routing_key,
            "event_action": "trigger",
            "payload": {
                "summary": summary,
                "source": source,
                "severity": severity,
                "custom_details": custom_details or {}
            }
        }

        try:
            response = requests.post("https://events.pagerduty.com/v2/enqueue", json=payload, timeout=5)
            response.raise_for_status()
            logger.info(f"PagerDuty alert triggered: {summary}")
        except Exception as e:
            logger.error(f"Failed to trigger PagerDuty alert: {str(e)}")

# Global instance
alert_manager = AlertManager()
