"""
Simple GPU health check utility.
Checks /health returns device=cuda and use_gpu=true.
"""
import sys
import json
import requests

def main():
    url = "http://127.0.0.1:5005/health"
    try:
        r = requests.get(url, timeout=3)
    except Exception as e:
        print(f"ERROR: Health request failed: {e}")
        sys.exit(1)
    if r.status_code != 200:
        print(f"ERROR: HTTP {r.status_code} at {url}")
        sys.exit(1)
    h = r.json()
    device = str(h.get("device"))
    use_gpu = bool(h.get("use_gpu"))
    if device == "cuda" and use_gpu is True:
        print("OK: GPU health is valid")
        sys.exit(0)
    else:
        print(f"ERROR: Invalid health: device={device}, use_gpu={use_gpu}")
        sys.exit(1)

if __name__ == "__main__":
    main()
