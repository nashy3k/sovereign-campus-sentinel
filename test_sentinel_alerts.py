"""Run with: python -m unittest test_sentinel_alerts -v  (stdlib only, no hardware needed)."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs

import sentinel_alerts as sa


class ParseFightVerdictTests(unittest.TestCase):
    def test_yes_and_no(self):
        self.assertTrue(sa.parse_fight_verdict("Severity: HIGH\nFIGHT_DETECTED: YES"))
        self.assertFalse(sa.parse_fight_verdict("Severity: LOW\nFIGHT_DETECTED: NO"))

    def test_tolerates_markdown_case_and_spacing(self):
        self.assertTrue(sa.parse_fight_verdict("**FIGHT_DETECTED:** yes"))
        self.assertTrue(sa.parse_fight_verdict("fight_detected = YES."))
        self.assertIsNone(sa.parse_fight_verdict("Fight detected. No weapons seen."))  # prose, not the verdict line
        self.assertFalse(sa.parse_fight_verdict("`FIGHT_DETECTED: No`"))

    def test_last_verdict_wins(self):
        text = "Format is FIGHT_DETECTED: YES or NO.\nNo punches seen.\nFIGHT_DETECTED: NO"
        self.assertFalse(sa.parse_fight_verdict(text))

    def test_missing_verdict_is_none_not_a_guess(self):
        self.assertIsNone(sa.parse_fight_verdict("Two students are fighting near the fence."))
        self.assertIsNone(sa.parse_fight_verdict(""))
        self.assertIsNone(sa.parse_fight_verdict("FIGHT_DETECTED: MAYBE"))


class FakeDisplay(BaseHTTPRequestHandler):
    calls = []
    token = "s3cret"

    def _reply(self, code, body):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self._reply(200, {"ok": True, "alert": False, "count": 0}) if self.path == "/status" \
            else self._reply(404, {"ok": False})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        form = {k: v[0] for k, v in parse_qs(self.rfile.read(length).decode()).items()}
        if self.headers.get("X-Sentinel-Token") != self.token:
            return self._reply(401, {"ok": False, "error": "unauthorized"})
        FakeDisplay.calls.append((self.path, form))
        self._reply(200, {"ok": True})

    def log_message(self, *args):
        pass


class DisplayClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeDisplay)
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        FakeDisplay.calls.clear()

    def test_send_alert_posts_clip_and_severity(self):
        ok, err = sa.send_fight_alert(self.url, "s3cret", "gate.mp4", "HIGH")
        self.assertTrue(ok, err)
        path, form = FakeDisplay.calls[0]
        self.assertEqual(path, "/alert")
        self.assertEqual((form["clip"], form["severity"]), ("gate.mp4", "HIGH"))
        self.assertTrue(form["time"])

    def test_wrong_token_reports_rejection(self):
        ok, err = sa.send_fight_alert(self.url, "wrong", "gate.mp4")
        self.assertFalse(ok)
        self.assertIn("token", err)
        self.assertEqual(FakeDisplay.calls, [])

    def test_clear_and_status(self):
        self.assertTrue(sa.clear_display_alert(self.url, "s3cret")[0])
        self.assertEqual(FakeDisplay.calls[0][0], "/clear")
        status, err = sa.get_display_status(self.url)
        self.assertEqual(status["alert"], False, err)

    def test_unreachable_display_returns_error_instead_of_raising(self):
        ok, err = sa.send_fight_alert("http://127.0.0.1:1", "s3cret", "gate.mp4")
        self.assertFalse(ok)
        self.assertTrue(err)
        status, err = sa.get_display_status("http://127.0.0.1:1")
        self.assertIsNone(status)


if __name__ == "__main__":
    unittest.main()
