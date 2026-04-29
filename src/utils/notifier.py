"""
Webhook Notifier — sends real-time alerts to Slack or Discord.

Requires WEBHOOK_URL environment variable to be set.
If not set, messages are logged locally (dry-run mode).
"""

import os
import json
import requests
from datetime import datetime, timezone
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Notifier:
    """
    Sends real-time execution and risk alerts to a configured Webhook (Slack or Discord).
    Requires WEBHOOK_URL environment variable to be set.
    """
    
    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url or os.getenv("WEBHOOK_URL")
        
        # Simple detection if it's Discord to use Slack-compatible payloads or discord native
        self.is_discord = self.webhook_url and "discord.com" in self.webhook_url

    def send_message(self, title: str, message: str, level: str = "info"):
        """
        Send a formatted message.
        level: info, warning, error, success
        """
        if not self.webhook_url:
            logger.info(f"[NOTIFIER DRY-RUN] {title} | {message}")
            return

        colors = {
            "info": "#3498db",      # Blue
            "success": "#2ecc71",   # Green
            "warning": "#f1c40f",   # Yellow
            "error": "#e74c3c"      # Red
        }
        color = colors.get(level, colors["info"])
        
        try:
            if self.is_discord:
                payload = self._build_discord_payload(title, message, color)
            else:
                payload = self._build_slack_payload(title, message, color)
                
            response = requests.post(
                self.webhook_url, 
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")

    def _build_slack_payload(self, title: str, message: str, color: str) -> dict:
        return {
            "attachments": [
                {
                    "fallback": f"{title} - {message}",
                    "color": color,
                    "title": title,
                    "text": message,
                    "footer": "Chrono Quant AI",
                    "ts": int(datetime.now(timezone.utc).timestamp())
                }
            ]
        }
        
    def _build_discord_payload(self, title: str, message: str, color: str) -> dict:
        # Convert hex color to int for Discord
        int_color = int(color.replace("#", ""), 16)
        return {
            "embeds": [
                {
                    "title": title,
                    "description": message,
                    "color": int_color,
                    "footer": {"text": "Chrono Quant AI"},
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ]
        }
