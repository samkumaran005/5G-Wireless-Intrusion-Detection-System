"""
5G Wireless Intrusion Detection System (5G-WIDS)
HTSTCL-GNN Interactive Web Dashboard
"""
import os
import sys

# Ensure current directory is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dashboard.app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 70)
    print(" 5G WIRELESS INTRUSION DETECTION SYSTEM (5G-WIDS)")
    print(" Powered by HTSTCL-GNN (Deep Spatio-Temporal Graph Neural Network)")
    print("=" * 70)
    print(f" Dashboard URL : http://127.0.0.1:{port}")
    print(f" Navigation    : Dashboard | Live Traffic Upload & Threat Detection")
    print(f" Model Status  : htstcl_gnn_best.pth (98.41% Test Accuracy)")
    print("=" * 70 + "\n")
    app.run(host="127.0.0.1", port=port, debug=False)
