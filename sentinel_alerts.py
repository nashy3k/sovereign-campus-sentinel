#!/usr/bin/env python3
"""
Fight verdict parsing and the Waveshare ESP32-S3-Touch-LCD-5 alert display client.

No third-party or Streamlit imports, so it can be used from the dashboard, a script, or tests.
The display (firmware/esp32s3_alert_display) exposes POST /alert, POST /clear and GET /status
on the local network; every request carries the shared secret in `X-Sentinel-Token`.

CLI:  python sentinel_alerts.py status|test|clear [--url URL] [--token TOKEN]
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional, Tuple

DEFAULT_DISPLAY_URL = os.environ.get("SENTINEL_DISPLAY_URL", "http://sentinel-display.local")
DEFAULT_DISPLAY_TOKEN = os.environ.get("SENTINEL_DISPLAY_TOKEN", "")
DISPLAY_TIMEOUT_S = 4.0

# The model is asked to finish its answer with this line (see SOVEREIGN_SYSTEM_PROMPT in app.py).
FIGHT_VERDICT_INSTRUCTION = "FIGHT_DETECTED: YES or FIGHT_DETECTED: NO"
_FIGHT_VERDICT_RE = re.compile(r"FIGHT_DETECTED\W*(YES|NO)\b", re.IGNORECASE)


def parse_fight_verdict(text: str) -> Optional[bool]:
    """True/False from the model's final `FIGHT_DETECTED: YES|NO` line; None if it gave no verdict.

    The last match wins, so a model that quotes the format earlier in its answer is still read correctly.
    """
    matches = _FIGHT_VERDICT_RE.findall(text or "")
    return matches[-1].upper() == "YES" if matches else None


def _request(base_url: str, path: str, token: str, fields: Optional[dict] = None,
             method: str = "POST") -> Tuple[bool, dict, str]:
    """Return (ok, parsed JSON body, error message). Never raises."""
    url = base_url.rstrip("/") + path
    data = urllib.parse.urlencode(fields or {}).encode() if method == "POST" else None
    req = urllib.request.Request(url, data=data, method=method)
    if token:
        req.add_header("X-Sentinel-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=DISPLAY_TIMEOUT_S) as resp:
            body = json.loads(resp.read().decode() or "{}")
        return bool(body.get("ok", True)), body, ""
    except urllib.error.HTTPError as e:
        e.close()
        reason = "rejected token (check SENTINEL_DISPLAY_TOKEN)" if e.code == 401 else f"HTTP {e.code}"
        return False, {}, reason
    except Exception as e:  # URLError, timeout, bad JSON, unresolved .local name...
        return False, {}, str(getattr(e, "reason", e))


def send_fight_alert(base_url: str, token: str, clip: str, severity: str = "") -> Tuple[bool, str]:
    """Trigger the 'Fighting is detected' alert on the display. Returns (ok, error)."""
    ok, _, err = _request(base_url, "/alert", token, {
        "clip": clip, "severity": severity, "time": time.strftime("%d %b %Y %H:%M:%S"),
    })
    return ok, err


def clear_display_alert(base_url: str, token: str) -> Tuple[bool, str]:
    """Return the display to its monitoring screen. Returns (ok, error)."""
    ok, _, err = _request(base_url, "/clear", token)
    return ok, err


def get_display_status(base_url: str) -> Tuple[Optional[dict], str]:
    """Return (status JSON, error); status is None if the display is unreachable."""
    ok, body, err = _request(base_url, "/status", "", method="GET")
    return (body if ok else None), err


def main() -> int:
    parser = argparse.ArgumentParser(description="Control the Sentinel ESP32 alert display.")
    parser.add_argument("action", choices=["status", "test", "clear"])
    parser.add_argument("--url", default=DEFAULT_DISPLAY_URL)
    parser.add_argument("--token", default=DEFAULT_DISPLAY_TOKEN)
    args = parser.parse_args()

    if args.action == "status":
        status, err = get_display_status(args.url)
        print(json.dumps(status) if status else f"Display unreachable at {args.url}: {err}")
        return 0 if status else 1
    ok, err = (send_fight_alert(args.url, args.token, "TEST CLIP", "TEST") if args.action == "test"
               else clear_display_alert(args.url, args.token))
    print("OK" if ok else f"Failed: {err}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
