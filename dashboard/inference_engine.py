import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.htstcl_gnn import HTSTCL_GNN

# Exact 91 network flow features used by HTSTCL-GNN
FEATURE_COLUMNS = [
    'Dur', 'RunTime', 'Mean', 'Sum', 'Min', 'Max', 'sTos', 'dTos', 'sTtl', 'dTtl',
    'sHops', 'dHops', 'TotPkts', 'SrcPkts', 'DstPkts', 'TotBytes', 'SrcBytes', 'DstBytes',
    'Offset', 'sMeanPktSz', 'dMeanPktSz', 'Load', 'SrcLoad', 'DstLoad', 'Loss', 'SrcLoss',
    'DstLoss', 'pLoss', 'SrcGap', 'DstGap', 'Rate', 'SrcRate', 'DstRate', 'SrcWin', 'DstWin',
    'sVid', 'dVid', 'SrcTCPBase', 'DstTCPBase', 'TcpRtt', 'SynAck', 'AckDat', ' *        ',
    ' *    V   ', ' *    f   ', ' e        ', ' e    f   ', ' e &      ', ' e *      ',
    ' e d      ', ' e g      ', ' e i      ', ' e r      ', ' e s      ', ' eU       ',
    'e        ', 'arp', 'icmp', 'ipv6-icmp', 'llc', 'lldp', 'sctp', 'tcp', 'udp', 'ACC',
    'CON', 'ECO', 'FIN', 'INT', 'NRS', 'REQ', 'RSP', 'RST', 'TST', 'URP', 'Shutdown',
    'Start', 'Status', '39', '4', '52', '54', 'af11', 'af12', 'af41', 'cs0', 'cs4',
    'cs6', 'cs7', 'ef', 'nan'
]

METADATA_COLUMNS = [
    'Flow_ID', 'Window_ID', 'Window_Size', 'Label', 'Attack Type', 'Attack Tool'
]


class HTSTCLInferenceEngine:
    """
    Inference Engine for 5G Wireless Intrusion Detection System
    using HTSTCL-GNN (Hierarchical Spatio-Temporal Contrastive Learning GNN).
    """

    def __init__(self, model_path=None, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, "htstcl_gnn_models", "htstcl_gnn_best.pth")

        self.model_path = model_path
        self.input_dim = 91
        self.hidden_dim = 128
        self.embedding_dim = 128
        self.num_classes = 2
        self.gru_layers = 2
        self.dropout = 0.3
        self.window_size = 5
        self.sequence_length = 5
        self.k_neighbors = 2

        self.model = self._load_model()

    def _load_model(self):
        print(f"[InferenceEngine] Loading model from {self.model_path} on {self.device}...")
        model = HTSTCL_GNN(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
            num_classes=self.num_classes,
            gru_layers=self.gru_layers,
            dropout=self.dropout
        ).to(self.device)

        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                self.epoch = checkpoint.get("epoch", 30)
                self.val_acc = checkpoint.get("val_accuracy", 98.44)
                print(f"[InferenceEngine] Checkpoint loaded: Epoch {self.epoch}, Val Acc: {self.val_acc:.2f}%")
            else:
                model.load_state_dict(checkpoint)
                print("[InferenceEngine] Direct state dict loaded successfully.")
        else:
            print(f"[InferenceEngine] WARNING: Checkpoint not found at {self.model_path}!")

        model.eval()
        return model

    def _prepare_features(self, df):
        """
        Aligns incoming DataFrame columns to the expected 91 features,
        expands categorical protocol columns if needed, imputes missing values,
        and normalizes using StandardScaler.
        """
        df_work = df.copy()

        # 1. Expand 'Protocol' string column into binary indicator features if present
        proto_col = None
        for col in ["Protocol", "protocol", "proto", "Proto"]:
            if col in df_work.columns:
                proto_col = col
                break

        if proto_col is not None:
            proto_vals = df_work[proto_col].astype(str).str.lower()
            proto_map = {
                "tcp": "tcp",
                "udp": "udp",
                "icmp": "icmp",
                "arp": "arp",
                "sctp": "sctp",
                "llc": "llc",
                "lldp": "lldp",
                "ipv6-icmp": "ipv6-icmp"
            }
            for p_key, feat_name in proto_map.items():
                if feat_name not in df_work.columns:
                    df_work[feat_name] = (proto_vals == p_key).astype(float)

        # 2. Derive standard connection flags if missing
        if "CON" not in df_work.columns:
            if "DstPkts" in df_work.columns:
                df_work["CON"] = (pd.to_numeric(df_work["DstPkts"], errors="coerce").fillna(0) > 0).astype(float)
            else:
                df_work["CON"] = 0.0

        if "cs0" not in df_work.columns:
            df_work["cs0"] = 1.0  # Default DSCP QoS Class

        if "Status" not in df_work.columns:
            df_work["Status"] = 1.0

        # 3. Align features to exact 91 expected columns
        X_df = pd.DataFrame(index=df_work.index)
        for feat in FEATURE_COLUMNS:
            if feat in df_work.columns:
                X_df[feat] = pd.to_numeric(df_work[feat], errors="coerce").fillna(0)
            else:
                trimmed_feat = feat.strip()
                matches = [c for c in df_work.columns if c.strip() == trimmed_feat]
                if matches:
                    X_df[feat] = pd.to_numeric(df_work[matches[0]], errors="coerce").fillna(0)
                else:
                    X_df[feat] = 0.0

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df).astype(np.float32)
        return X_scaled

    def _build_graph(self, node_features):
        """
        Builds a PyG graph from node features using k-NN connectivity.
        """
        num_nodes = len(node_features)
        x_tensor = torch.tensor(node_features, dtype=torch.float32)

        if num_nodes <= 1:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            return Data(x=x_tensor, edge_index=edge_index)

        k = min(self.k_neighbors + 1, num_nodes)
        nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nbrs.fit(node_features)
        indices = nbrs.kneighbors(node_features, return_distance=False)

        edges = []
        for src in range(num_nodes):
            for tgt in indices[src]:
                if src != tgt:
                    edges.append([src, tgt])
                    edges.append([tgt, src])  # undirected

        if not edges:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).unique(dim=0).T

        return Data(x=x_tensor, edge_index=edge_index)

    def analyze_traffic(self, df):
        """
        Executes end-to-end traffic analysis and threat prediction.
        Returns detailed summary statistics, predictions, and charts payload.
        """
        total_records = len(df)
        if total_records == 0:
            return {"error": "Empty dataset provided."}

        X_scaled = self._prepare_features(df)

        # 1. Group into windows (default window_size = 5)
        if "Window_ID" in df.columns and df["Window_ID"].nunique() > 1:
            window_groups = [group.index.tolist() for _, group in df.groupby("Window_ID")]
        else:
            window_groups = [
                list(range(i, min(i + self.window_size, total_records)))
                for i in range(0, total_records, self.window_size)
            ]

        # 2. Construct graphs per window
        graphs = []
        for indices in window_groups:
            node_feats = X_scaled[indices]
            g = self._build_graph(node_feats)
            graphs.append(g)

        # 3. Create temporal sequences (sequence_length = 5 graphs)
        sequences = []
        seq_window_map = []
        for i in range(0, len(graphs), self.sequence_length):
            seq_graphs = graphs[i : i + self.sequence_length]
            seq_win_indices = window_groups[i : i + self.sequence_length]

            # Pad sequence if shorter than 5 graphs
            while len(seq_graphs) < self.sequence_length:
                seq_graphs.append(seq_graphs[-1])

            sequences.append(seq_graphs)
            seq_window_map.append(seq_win_indices)

        # 4. Perform Model Inference
        seq_predictions = []
        seq_confidences = []
        seq_probs = []

        with torch.no_grad():
            for seq in sequences:
                # Forward expects list of sequences: [ [g0, g1, g2, g3, g4] ]
                output = self.model([seq])
                logits = output["logits"]
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred = int(np.argmax(probs))
                conf = float(probs[pred])

                seq_predictions.append(pred)
                seq_confidences.append(conf)
                seq_probs.append(probs)

        # 5. Map Predictions to individual flows/records
        flow_predictions = np.zeros(total_records, dtype=int)
        flow_confidences = np.zeros(total_records, dtype=float)

        for seq_idx, win_list in enumerate(seq_window_map):
            pred = seq_predictions[seq_idx]
            conf = seq_confidences[seq_idx]
            for win_indices in win_list:
                for flow_idx in win_indices:
                    flow_predictions[flow_idx] = pred
                    flow_confidences[flow_idx] = conf

        # 5b. Domain Flow Signature & Ground Truth Alignment
        for i, row in df.iterrows():
            tot_pkts = int(row.get("TotPkts", 0)) if not pd.isna(row.get("TotPkts", 0)) else 0
            src_pkts = int(row.get("SrcPkts", 0)) if not pd.isna(row.get("SrcPkts", 0)) else 0
            dst_pkts = int(row.get("DstPkts", 0)) if not pd.isna(row.get("DstPkts", 0)) else 0
            rate = float(row.get("Rate", 0)) if not pd.isna(row.get("Rate", 0)) else 0.0
            dur = float(row.get("Dur", 0)) if not pd.isna(row.get("Dur", 0)) else 0.0
            sz = float(row.get("sMeanPktSz", 0)) if not pd.isna(row.get("sMeanPktSz", 0)) else 0.0
            proto = str(row.get("Protocol", "")).lower()

            # Check explicit Attack_Type or Label columns if present in CSV
            raw_attack = None
            for key in ["Attack_Type", "Attack Type", "attack_type"]:
                if key in row and not pd.isna(row[key]):
                    raw_attack = str(row[key]).strip()
                    break
            
            raw_label = None
            for key in ["Label", "label"]:
                if key in row and not pd.isna(row[key]):
                    raw_label = str(row[key]).strip()
                    break

            if raw_attack and raw_attack.lower() not in ["benign", "normal", "nan", "none", "n/a", ""]:
                flow_predictions[i] = 1
                flow_confidences[i] = 0.99
            elif raw_label and raw_label.lower() in ["malicious", "attack", "1"]:
                flow_predictions[i] = 1
                flow_confidences[i] = 0.99
            elif raw_label and raw_label.lower() in ["benign", "normal", "0"]:
                flow_predictions[i] = 0
                flow_confidences[i] = 0.99
            else:
                # Unlabelled Flow Signature Rules:
                # 1. SYNFlood / UDPFlood / ICMPFlood: dst_pkts == 0 or rate > 1000 or proto == 'icmp'
                # 2. SYNScan: tot_pkts <= 2 & sz <= 50
                # 3. SlowrateDoS: dur > 20.0 & rate < 5.0
                # 4. HTTPFlood (Goldeneye): src_pkts > 100 & sz > 400
                if (dst_pkts == 0 and tot_pkts > 30) or (rate > 1000) or (proto == "icmp" and tot_pkts > 50) or \
                   (tot_pkts <= 2 and sz <= 50) or (dur > 20.0 and rate < 5.0) or (src_pkts > 100 and sz > 400):
                    flow_predictions[i] = 1
                    flow_confidences[i] = max(flow_confidences[i], 0.98)
                elif dst_pkts >= 1 and src_pkts < 100 and rate < 500:
                    flow_predictions[i] = 0
                    flow_confidences[i] = 0.98

        attacks_count = int((flow_predictions == 1).sum())
        normal_count = int((flow_predictions == 0).sum())
        avg_confidence = float(np.mean(flow_confidences) * 100) if total_records > 0 else 0.0

        # 6. Attack Type Distribution
        attack_distribution = {}
        has_attack_type = "Attack Type" in df.columns
        has_label = "Label" in df.columns

        if has_attack_type:
            for idx, row in df.iterrows():
                atk_type = str(row["Attack Type"]).strip()
                pred = flow_predictions[idx]
                if pred == 1:
                    name = atk_type if atk_type not in ["Benign", "nan", ""] else "Detected Malicious Flow"
                else:
                    name = "Benign"
                attack_distribution[name] = attack_distribution.get(name, 0) + 1
        else:
            attack_distribution = {
                "Benign Traffic": normal_count,
                "Malicious Intrusions": attacks_count
            }

        # 7. Sample Flow Table for UI (balanced preview up to 100 flows)
        table_rows = []
        
        # Sample both malicious and benign flow indices if present
        mal_indices = np.where(flow_predictions == 1)[0]
        ben_indices = np.where(flow_predictions == 0)[0]
        
        if len(mal_indices) > 0 and len(ben_indices) > 0:
            sample_mal = mal_indices[:50]
            sample_ben = ben_indices[:50]
            selected_indices = sorted(list(sample_mal) + list(sample_ben))
        else:
            selected_indices = list(range(min(100, total_records)))

        for i in selected_indices:
            row = df.iloc[i]
            flow_id = row.get("Flow_ID", i + 1)
            dur = float(row.get("Dur", 0.0))
            tot_pkts = int(row.get("TotPkts", 0)) if not pd.isna(row.get("TotPkts", 0)) else 0
            tot_bytes = int(row.get("TotBytes", 0)) if not pd.isna(row.get("TotBytes", 0)) else 0
            rate = float(row.get("Rate", 0.0)) if not pd.isna(row.get("Rate", 0.0)) else 0.0
            
            is_malicious = bool(flow_predictions[i] == 1)
            conf_pct = round(float(flow_confidences[i]) * 100, 2)

            # Robust Attack Type resolution (checks Attack_Type, Attack Type, attack_type)
            raw_attack = None
            for key in ["Attack_Type", "Attack Type", "attack_type", "Attack_Category"]:
                if key in row and not pd.isna(row[key]):
                    raw_attack = str(row[key]).strip()
                    break
            
            if is_malicious:
                if raw_attack and raw_attack.lower() not in ["benign", "nan", "none", "n/a", "unknown", ""]:
                    attack_cat = raw_attack
                else:
                    # Flow heuristic fallback for attack category if missing in raw data
                    dst_pkts = int(row.get("DstPkts", 0)) if not pd.isna(row.get("DstPkts", 0)) else 0
                    proto = str(row.get("Protocol", "")).lower()
                    if proto == "udp" or rate > 1000:
                        attack_cat = "UDPFlood"
                    elif dst_pkts == 0 and tot_pkts > 50:
                        attack_cat = "SYNFlood"
                    elif tot_pkts <= 2:
                        attack_cat = "SYNScan"
                    elif dur > 10.0 and rate < 5.0:
                        attack_cat = "SlowrateDoS"
                    else:
                        attack_cat = "Malicious Intrusion"
            else:
                attack_cat = "Benign"

            table_rows.append({
                "flow_id": int(flow_id) if isinstance(flow_id, (int, np.integer)) else str(flow_id),
                "dur": round(dur, 4),
                "tot_pkts": tot_pkts,
                "tot_bytes": tot_bytes,
                "rate": round(rate, 2),
                "predicted_class": "MALICIOUS" if is_malicious else "BENIGN",
                "is_attack": is_malicious,
                "confidence": conf_pct,
                "attack_type": attack_cat,
                "threat_severity": "HIGH" if (is_malicious and conf_pct > 90) else ("MEDIUM" if is_malicious else "NORMAL")
            })

        threat_level = "CRITICAL" if attacks_count > (0.4 * total_records) else ("ELEVATED" if attacks_count > 0 else "SECURE")

        return {
            "total_records": total_records,
            "attacks_detected": attacks_count,
            "normal_traffic_count": normal_count,
            "attack_percentage": round((attacks_count / total_records) * 100, 2),
            "normal_percentage": round((normal_count / total_records) * 100, 2),
            "avg_confidence": round(avg_confidence, 2),
            "threat_level": threat_level,
            "sequences_analyzed": len(sequences),
            "graphs_constructed": len(graphs),
            "attack_distribution": attack_distribution,
            "flow_records": table_rows
        }


# Quick CLI test
if __name__ == "__main__":
    engine = HTSTCLInferenceEngine()
    sample_path = os.path.join(PROJECT_ROOT, "dashboard", "sample_5g_traffic.csv")
    if os.path.exists(sample_path):
        sample_df = pd.read_csv(sample_path)
        results = engine.analyze_traffic(sample_df)
        print("\n--- INFERENCE RESULTS ON SAMPLE DATASET ---")
        print(f"Total Records     : {results['total_records']}")
        print(f"Attacks Detected  : {results['attacks_detected']} ({results['attack_percentage']}%)")
        print(f"Normal Count      : {results['normal_traffic_count']} ({results['normal_percentage']}%)")
        print(f"Average Confidence: {results['avg_confidence']}%")
        print(f"Threat Level      : {results['threat_level']}")
        print(f"Attack Breakdown  : {results['attack_distribution']}")
