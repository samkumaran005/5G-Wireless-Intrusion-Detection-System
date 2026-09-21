import os
import sys
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dashboard.inference_engine import HTSTCLInferenceEngine, FEATURE_COLUMNS

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

# Initialize Inference Engine (cached singleton)
print("[Flask] Initializing HTSTCL-GNN Inference Engine...")
engine = HTSTCLInferenceEngine()

# Model Metadata & Evaluation Summary
MODEL_METADATA = {
    "name": "HTSTCL-GNN",
    "full_title": "Hierarchical Spatio-Temporal Contrastive Learning Graph Neural Network",
    "version": "1.0-production",
    "architecture": {
        "spatial_encoder": "Graph Attention Network (GAT) with 4 Attention Heads",
        "temporal_encoder": "2-Layer Bidirectional Gated Recurrent Unit (Bi-GRU)",
        "fusion_mechanism": "Hierarchical Self-Attention Cross-Modal Fusion",
        "self_supervised_loss": "InfoNCE Spatio-Temporal Contrastive Loss",
        "input_features": 91,
        "hidden_dim": 128,
        "embedding_dim": 128,
        "num_classes": 2,
        "window_size": 5,
        "sequence_length": 5
    },
    "metrics": {
        "accuracy": 98.41,
        "precision": 98.83,
        "recall": 98.66,
        "f1_score": 98.75,
        "roc_auc": 0.9986,
        "avg_precision": 0.9993,
        "best_epoch": 30,
        "val_accuracy": 98.44,
        "val_loss": 0.0402,
        "confusion_matrix": {
            "true_benign": 2607,
            "false_malicious": 54,
            "false_benign": 62,
            "true_malicious": 4573
        }
    },
    "dataset": {
        "name": "5G-IDS Wireless Flow Dataset",
        "total_test_samples": 7296,
        "benign_support": 2661,
        "malicious_support": 4635
    }
}

# Feature Categorization for UI Explorer
FEATURE_GROUPS = {
    "Temporal & Volume": [
        "Dur", "RunTime", "Mean", "Sum", "Min", "Max", "TotPkts", "SrcPkts",
        "DstPkts", "TotBytes", "SrcBytes", "DstBytes", "Offset", "sMeanPktSz", "dMeanPktSz"
    ],
    "Network Rates & Load": [
        "Load", "SrcLoad", "DstLoad", "Rate", "SrcRate", "DstRate"
    ],
    "TCP Flow & Connection Dynamics": [
        "sTos", "dTos", "sTtl", "dTtl", "sHops", "dHops", "Loss", "SrcLoss",
        "DstLoss", "pLoss", "SrcGap", "DstGap", "SrcWin", "DstWin", "sVid",
        "dVid", "SrcTCPBase", "DstTCPBase", "TcpRtt", "SynAck", "AckDat"
    ],
    "Protocols & Layer-2/3 Identifiers": [
        "arp", "icmp", "ipv6-icmp", "llc", "lldp", "sctp", "tcp", "udp"
    ],
    "Connection Flags & State Machine": [
        "ACC", "CON", "ECO", "FIN", "INT", "NRS", "REQ", "RSP", "RST", "TST", "URP",
        "Shutdown", "Start", "Status"
    ],
    "QoS Slicing & DSCP Classes": [
        "39", "4", "52", "54", "af11", "af12", "af41", "cs0", "cs4", "cs6", "cs7", "ef", "nan"
    ]
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/model_info", methods=["GET"])
def get_model_info():
    return jsonify(MODEL_METADATA)


@app.route("/api/features", methods=["GET"])
def get_features():
    return jsonify({
        "total_features": len(FEATURE_COLUMNS),
        "groups": FEATURE_GROUPS,
        "feature_list": FEATURE_COLUMNS
    })


@app.route("/api/sample", methods=["GET"])
def get_sample_preview():
    sample_path = os.path.join(PROJECT_ROOT, "dashboard", "sample_5g_traffic.csv")
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        preview_rows = df.head(10).to_dict(orient="records")
        return jsonify({
            "total_rows": len(df),
            "columns": list(df.columns),
            "preview": preview_rows
        })
    return jsonify({"error": "Sample file not found."}), 404


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        if request.is_json and request.json.get("use_sample"):
            sample_path = os.path.join(PROJECT_ROOT, "dashboard", "sample_5g_traffic.csv")
            if not os.path.exists(sample_path):
                return jsonify({"error": "Sample traffic dataset not found."}), 404
            df = pd.read_csv(sample_path)
            results = engine.analyze_traffic(df)
            return jsonify(results)

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded. Please upload a CSV file."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file."}), 400

        if not file.filename.lower().endswith(".csv"):
            return jsonify({"error": "Invalid format. Only .CSV network traffic files are supported."}), 400

        df = pd.read_csv(file)
        results = engine.analyze_traffic(df)
        return jsonify(results)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/download/sample", methods=["GET"])
def download_sample():
    sample_path = os.path.join(PROJECT_ROOT, "dashboard", "sample_5g_traffic.csv")
    if os.path.exists(sample_path):
        return send_file(
            sample_path,
            as_attachment=True,
            download_name="sample_5g_traffic.csv",
            mimetype="text/csv"
        )
    return jsonify({"error": "File not found."}), 404


@app.route("/assets/<path:filename>")
def serve_project_assets(filename):
    """
    Serves generated plots and artifacts from evaluation_results/ or project root.
    """
    eval_dir = os.path.join(PROJECT_ROOT, "evaluation_results")
    if os.path.exists(os.path.join(eval_dir, filename)):
        return send_from_directory(eval_dir, filename)

    if os.path.exists(os.path.join(PROJECT_ROOT, filename)):
        return send_from_directory(PROJECT_ROOT, filename)

    return jsonify({"error": f"Asset '{filename}' not found."}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f"5G WIRELESS INTRUSION DETECTION SYSTEM (HTSTCL-GNN)")
    print(f"Server starting on http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False)
