#!/usr/bin/env python3
"""
Check if observability routes are available on the running server.
"""

import sys
import requests
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_routes():
    """Check if observability routes are available."""
    base_url = "http://localhost:8000"
    
    routes_to_check = [
        ("/observability", "GET"),
        ("/observability/status", "GET"),
        ("/ws/observability", "WEBSOCKET"),  # Can't easily test WebSocket with requests
    ]
    
    print("=" * 60)
    print("Checking Observability Routes")
    print("=" * 60)
    
    for route, method in routes_to_check:
        url = f"{base_url}{route}"
        try:
            if method == "WEBSOCKET":
                print(f"{route:30} - WebSocket (manual check needed)")
                continue
            
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"{route:30} - ✓ Available (200 OK)")
            elif response.status_code == 404:
                print(f"{route:30} - ✗ Not Found (404) - Server needs restart")
            else:
                print(f"{route:30} - ⚠ Status: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"{route:30} - ✗ Connection Error - Server not running")
        except Exception as e:
            print(f"{route:30} - ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("To fix: Restart the server with:")
    print("  cd /Users/mac/Desktop/Dev/carebot_dev/patient-ai-v3")
    print("  python run.py")
    print("=" * 60)

if __name__ == "__main__":
    check_routes()

