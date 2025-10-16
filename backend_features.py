# backend_features.py (Flask)
from flask import Flask, request, jsonify
import os, requests, math
from decimal import Decimal

app = Flask(__name__)

ETHERSCAN_API_KEY = os.environ.get("ETHERSCAN_API_KEY", "YOUR_API_KEY_HERE")
ETHERSCAN_BASE = "https://api.etherscan.io/api"

def wei_to_eth(wei_str):
    try:
        return float(Decimal(wei_str) / Decimal(10**18))
    except:
        return 0.0

@app.route("/api/extract_features", methods=["GET"])
def extract_features():
    """
    Returns raw features for a single wallet address as a 2D list
    Example output:
      {"features": [[tx_count, incoming_count, outgoing_count, total_in_eth, total_out_eth, avg_tx_value, unique_counterparties, erc20_count, avg_gas_price_gwei, is_contract]]}
    IMPORTANT: change ordering/columns to match preprocessor used in training.
    """
    address = request.args.get("wallet", "").strip()
    if not address:
        return jsonify({"error": "Missing wallet query parameter"}), 400

    try:
        # 1) Normal transactions
        params = {
            "module": "account",
            "action": "txlist",
            "address": address,
            "startblock": 0,
            "endblock": 99999999,
            "sort": "asc",
            "apikey": ETHERSCAN_API_KEY
        }
        r = requests.get(ETHERSCAN_BASE, params=params, timeout=15)
        tx_resp = r.json()
        txs = tx_resp.get("result", []) if tx_resp.get("status") in ("1","0") else []

        # 2) ERC20 token transfers
        params_tok = {
            "module": "account",
            "action": "tokentx",
            "address": address,
            "startblock": 0,
            "endblock": 99999999,
            "sort": "asc",
            "apikey": ETHERSCAN_API_KEY
        }
        r2 = requests.get(ETHERSCAN_BASE, params=params_tok, timeout=15)
        tok_resp = r2.json()
        tok_txs = tok_resp.get("result", []) if tok_resp.get("status") in ("1","0") else []

        # Basic numeric features (example)
        tx_count = len(txs)
        incoming_count = sum(1 for t in txs if t.get("to","").lower() == address.lower())
        outgoing_count = sum(1 for t in txs if t.get("from","").lower() == address.lower())
        total_in_eth = sum(wei_to_eth(t.get("value","0")) for t in txs if t.get("to","").lower() == address.lower())
        total_out_eth = sum(wei_to_eth(t.get("value","0")) for t in txs if t.get("from","").lower() == address.lower())
        avg_tx_value = ( (total_in_eth + total_out_eth) / tx_count ) if tx_count>0 else 0.0
        unique_counterparties = len(set(
            [t.get("from","").lower() for t in txs if t.get("from","").lower() != address.lower()] +
            [t.get("to","").lower() for t in txs if t.get("to","").lower() != address.lower()]
        ))

        # ERC20 related
        erc20_count = len(tok_txs)
        unique_tokens = len({t.get("contractAddress","").lower() for t in tok_txs if t.get("contractAddress")})
        # gas price average (in gwei): gasPrice is in wei
        gas_prices = [int(t.get("gasPrice", "0")) for t in txs if t.get("gasPrice")]
        avg_gas_price_gwei = (sum(gas_prices) / len(gas_prices) / 1e9) if gas_prices else 0.0

        # Determine if address is a contract via Etherscan getabi (quick check)
        # If ABI exists -> contract, otherwise EOA (rate-limited; optional)
        is_contract = 0
        try:
            abi_params = {"module": "contract", "action": "getabi", "address": address, "apikey": ETHERSCAN_API_KEY}
            r3 = requests.get(ETHERSCAN_BASE, params=abi_params, timeout=10)
            parsed = r3.json()
            if parsed.get("result") and parsed.get("result") != "Contract source code not verified":
                is_contract = 1
        except:
            is_contract = 0

        # Compose feature vector - IMPORTANT: order must match your training X columns
        features = [
          tx_count,
          incoming_count,
          outgoing_count,
          total_in_eth,
          total_out_eth,
          avg_tx_value,
          unique_counterparties,
          erc20_count,
          unique_tokens,
          avg_gas_price_gwei,
          is_contract
        ]

        # Wrap as 2D list (one sample)
        return jsonify({"features": [features]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
