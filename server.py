"""
Standalone Local HTTP Server for the Emergency Triage CDSS Prototype.
Zero external pip dependencies required (runs on built-in standard library).
"""

import sys
import json
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path

# Add prototype directory to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))

from cdss_prototype.engine import evaluate_patient_payload

PORT = 8000

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class CDSSRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR / 'static'), **kwargs)

    def do_POST(self):
        if self.path == '/api/predict':
            content_length = int(self.headers.get('Content-Length', 0))
            post_body = self.rfile.read(content_length)
            try:
                data = json.loads(post_body.decode('utf-8'))
                result = evaluate_patient_payload(data)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error', 'message': str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == '/api/sample_cases':
            samples = {
                "case1_high_risk": {
                    "title": "Case 1: Elderly Delayed Presentation (Severe Rupture Risk)",
                    "payload": {
                        "age": 68, "gender": 1, "bt": 38.7, "sbp": 105, "dbp": 65,
                        "hr": 108, "rr": 24, "time_hours": 36, "wbc": 16800, "pmn": 89
                    }
                },
                "case2_low_risk": {
                    "title": "Case 2: Young Early Appendicitis (Uncomplicated Phlegmonous)",
                    "payload": {
                        "age": 23, "gender": 0, "bt": 37.1, "sbp": 118, "dbp": 76,
                        "hr": 78, "rr": 18, "time_hours": 8, "wbc": 11200, "pmn": 72
                    }
                },
                "case3_data_corruption": {
                    "title": "Case 3: Data-Quality Guard Test (Inverted BP & RR Outlier)",
                    "payload": {
                        "age": 52, "gender": 1, "bt": 43.5, "sbp": 80, "dbp": 95,
                        "hr": 115, "rr": 75, "time_hours": 28, "wbc": 14000, "pmn": 82
                    }
                }
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(samples).encode('utf-8'))
        elif self.path == '/' or self.path == '':
            self.path = '/index.html'
            return super().do_GET()
        else:
            return super().do_GET()


def run_server(open_browser=True):
    server_address = ('127.0.0.1', PORT)
    httpd = ThreadedHTTPServer(server_address, CDSSRequestHandler)
    print(f"============================================================")
    print(f"  Emergency Triage CDSS Prototype Server Running!")
    print(f"  Local URL: http://127.0.0.1:{PORT}")
    print(f"  Engine Mode: Two-Tier (Zero-Lab Bedside + Comprehensive)")
    print(f"  Press Ctrl+C to terminate.")
    print(f"============================================================")
    if open_browser:
        webbrowser.open(f"http://127.0.0.1:{PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        httpd.server_close()

if __name__ == '__main__':
    open_b = '--no-browser' not in sys.argv
    run_server(open_browser=open_b)
