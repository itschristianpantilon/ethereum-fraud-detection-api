# flask_gnn_api_risk.py
from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, BatchNorm
import joblib
import numpy as np

# ============================================================
# ⚙️ Flask App
# ============================================================
app = Flask(__name__)

# ============================================================
# 🔧 Load Preprocessor
# ============================================================
preprocessor = joblib.load("preprocessor.joblib")

# ============================================================
# 🔧 Define GCN Model Class
# ============================================================
class GCN_FineTuned(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.4):
        super(GCN_FineTuned, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels // 2)
        self.bn2 = BatchNorm(hidden_channels // 2)
        self.conv3 = GCNConv(hidden_channels // 2, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x

# ============================================================
# 🔧 Load Model
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_X = np.zeros((1, preprocessor.transformers_[0][1].mean_.shape[0] + 
                        sum(len(cats) for cats in preprocessor.transformers_[1][1].categories_)))
model = GCN_FineTuned(in_channels=sample_X.shape[1], hidden_channels=256, out_channels=2).to(device)
model.load_state_dict(torch.load("model/best_gnn_model.pth", map_location=device))
model.eval()

# ============================================================
# 🔧 Utility: Create edge_index for batch
# ============================================================
def create_edge_index(num_nodes):
    if num_nodes == 1:
        return torch.tensor([[0], [0]], dtype=torch.long)
    edges = torch.randint(0, num_nodes, (2, min(num_nodes*5, 30000)))
    return edges

# ============================================================
# 🔧 Utility: Map probability to risk level
# ============================================================
def risk_level(prob_fraud):
    if prob_fraud < 0.3:
        return "Low"
    elif prob_fraud < 0.7:
        return "Medium"
    else:
        return "High"

# ============================================================
# 🔧 Prediction Endpoint with Risk
# ============================================================
@app.route("/predict_risk", methods=["POST"])
def predict_risk():
    try:
        json_data = request.json
        if "features" not in json_data:
            return jsonify({"error": "Missing 'features' in request"}), 400

        X_input = np.array(json_data["features"])  # shape: [num_samples, num_features]
        if len(X_input.shape) != 2:
            return jsonify({"error": "Features must be a 2D list"}), 400

        # Preprocess
        X_proc = preprocessor.transform(X_input)
        X_tensor = torch.tensor(X_proc.toarray(), dtype=torch.float).to(device)

        # Synthetic edges for batch
        edge_index = create_edge_index(X_tensor.shape[0]).to(device)
        data = Data(x=X_tensor, edge_index=edge_index)

        # Predict
        with torch.no_grad():
            out = model(data)
            probs = F.softmax(out, dim=1).cpu().numpy()  # [num_samples, 2]
            preds = out.argmax(dim=1).cpu().numpy()      # [num_samples]

        # Format output with risk level
        results = []
        for i in range(len(preds)):
            prob_fraud = probs[i][1]  # probability of fraud class
            results.append({
                "transaction_index": i,
                "pred_class": int(preds[i]),
                "probabilities": probs[i].tolist(),
                "risk_level": risk_level(prob_fraud)
            })

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================
# 🔧 Run Flask
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
