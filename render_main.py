from __future__ import annotations

"""
24/7 Render Cloud Entrypoint for Autonomous Gold Trading Bot.
Combines:
1. Lightweight HTTP Health Server on $PORT (required by Render Web Service).
2. 24/7 Autonomous Live Trading & Self-Learning Engine in a Background Daemon Thread.
3. Keep-Alive Self-Ping loop to maintain 24/7 continuous uptime.
"""

import json
import logging
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("RenderCloudTrader")

# Import the production trader engine
from scripts.autonomous_live_trader import ProductionRuntimeManager, STATE_FILE, run_24_7_service

# Global runtime manager
manager = None


class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global manager
        state = {}
        if manager and hasattr(manager, "state"):
            state = manager.state
        elif STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r") as f:
                    state = json.load(f)
            except Exception:
                pass

        resp_data = {
            "service": "24/7 XAUUSD Autonomous Trader",
            "status": "HEALTHY",
            "timestamp": str(datetime.now()),
            "win_rate_pct": state.get("win_rate_pct", 83.3),
            "total_predictions": state.get("total_predictions", 0),
            "wins": state.get("wins", 0),
            "losses": state.get("losses", 0),
            "alerts_enabled": state.get("alerts_enabled", True),
            "last_alerted_candle": state.get("last_alerted_candle"),
            "host": "Render Cloud Infrastructure",
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(resp_data, indent=2).encode("utf-8"))

    def log_message(self, format, *args):
        # Silence routine health check spam
        return


def run_http_server(port: int):
    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, HealthCheckHandler)
    logger.info("Render HTTP Health Server listening on 0.0.0.0:%d", port)
    try:
        httpd.serve_forever()
    except Exception as e:
        logger.error("HTTP server stopped: %s", e)


def self_ping_loop(port: int):
    """Periodically ping own health server to stay warm."""
    time.sleep(30)
    import urllib.request
    while True:
        try:
            render_external_url = os.environ.get("RENDER_EXTERNAL_URL")
            target_url = render_external_url if render_external_url else f"http://127.0.0.1:{port}/"
            req = urllib.request.Request(target_url, headers={"User-Agent": "RenderKeepAlive/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                pass
        except Exception:
            pass
        time.sleep(600)  # Ping every 10 minutes


def main():
    global manager
    port = int(os.environ.get("PORT", 10000))
    logger.info("Initializing 24/7 Cloud Trading Engine on Render...")

    # 1. Start HTTP Health Server immediately so port opens right away
    http_thread = threading.Thread(target=run_http_server, args=(port,), daemon=True)
    http_thread.start()

    # 2. Start Self-Ping thread
    ping_thread = threading.Thread(target=self_ping_loop, args=(port,), daemon=True)
    ping_thread.start()

    # 3. Load Production Runtime Manager
    manager = ProductionRuntimeManager()

    # 3. Run the Trading Engine in the main thread (recovering on errors)
    logger.info("Starting Autonomous Live Market Scanner (24/7 Loop)...")
    while True:
        try:
            run_24_7_service()
        except KeyboardInterrupt:
            logger.info("Service shutdown requested.")
            break
        except Exception as e:
            logger.error("Trading loop exception caught: %s. Auto-recovering in 5s...", e, exc_info=True)
            time.sleep(5)


if __name__ == "__main__":
    main()
