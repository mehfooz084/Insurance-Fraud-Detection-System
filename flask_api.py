"""
Insurance Fraud Detection System - Flask API
============================================
Standalone REST API for the HTML/CSS/JS frontend.
Run alongside (or instead of) the Streamlit app.py -- does NOT modify it.

Usage:
    python flask_api.py

Endpoints:
    GET  /api/sample-claim   ->  Returns a random claim from the CSV dataset
    POST /api/predict         ->  Runs the Decision Tree + returns prediction + reasons
    GET  /api/health          ->  Shows whether model/dataset loaded correctly
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # allow requests from file:// and localhost frontends



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(BASE_DIR, filename)
# ── Feature definitions (identical to app.py) ─────────────────────────────────
CATEGORICAL_FEATURES = [
    "policy_state", "policy_csl", "insured_sex", "insured_education_level",
    "insured_occupation", "insured_hobbies", "insured_relationship",
    "incident_type", "collision_type", "incident_severity",
    "authorities_contacted", "incident_state", "incident_city",
    "property_damage", "police_report_available", "auto_make", "auto_model"
]

NUMERICAL_FEATURES = [
    "months_as_customer", "age", "policy_deductable", "policy_annual_premium",
    "umbrella_limit", "capital-gains", "capital-loss", "incident_hour_of_the_day",
    "number_of_vehicles_involved", "bodily_injuries", "witnesses",
    "total_claim_amount", "injury_claim", "property_claim", "vehicle_claim",
    "auto_year", "days_to_incident"
]

ALL_FEATURES = [
    "months_as_customer", "age", "policy_state", "policy_csl", "policy_deductable",
    "policy_annual_premium", "umbrella_limit", "insured_sex", "insured_education_level",
    "insured_occupation", "insured_hobbies", "insured_relationship", "capital-gains",
    "capital-loss", "incident_type", "collision_type", "incident_severity",
    "authorities_contacted", "incident_state", "incident_city", "incident_hour_of_the_day",
    "number_of_vehicles_involved", "property_damage", "bodily_injuries", "witnesses",
    "police_report_available", "total_claim_amount", "injury_claim", "property_claim",
    "vehicle_claim", "auto_make", "auto_model", "auto_year", "days_to_incident"
]

FEATURE_DISPLAY_NAMES = {
    "months_as_customer":          "Months as Customer",
    "age":                         "Age",
    "policy_state":                "Policy State",
    "policy_csl":                  "Policy CSL",
    "policy_deductable":           "Policy Deductible",
    "policy_annual_premium":       "Annual Premium",
    "umbrella_limit":              "Umbrella Limit",
    "insured_sex":                 "Insured Sex",
    "insured_education_level":     "Education Level",
    "insured_occupation":          "Occupation",
    "insured_hobbies":             "Hobbies",
    "insured_relationship":        "Relationship",
    "capital-gains":               "Capital Gains",
    "capital-loss":                "Capital Loss",
    "incident_type":               "Incident Type",
    "collision_type":              "Collision Type",
    "incident_severity":           "Incident Severity",
    "authorities_contacted":       "Authorities Contacted",
    "incident_state":              "Incident State",
    "incident_city":               "Incident City",
    "incident_hour_of_the_day":    "Incident Hour",
    "number_of_vehicles_involved": "Vehicles Involved",
    "property_damage":             "Property Damage",
    "bodily_injuries":             "Bodily Injuries",
    "witnesses":                   "Witnesses",
    "police_report_available":     "Police Report Available",
    "total_claim_amount":          "Total Claim Amount",
    "injury_claim":                "Injury Claim",
    "property_claim":              "Property Claim",
    "vehicle_claim":               "Vehicle Claim",
    "auto_make":                   "Auto Make",
    "auto_model":                  "Auto Model",
    "auto_year":                   "Auto Year",
    "days_to_incident":            "Days to Incident",
}

MONEY_FIELDS = {
    "total_claim_amount", "injury_claim", "property_claim", "vehicle_claim",
    "policy_annual_premium", "umbrella_limit", "capital-gains", "capital-loss",
    "policy_deductable"
}

# ── Load artifacts at startup ──────────────────────────────────────────────────
print("[Flask API] Loading model and data...")

try:
    model = joblib.load(os.path.join(BASE_DIR, "decision_tree_model.pkl"))
    print("[Flask API] Model loaded OK")
except FileNotFoundError:
    model = None
    print("[Flask API] WARNING: decision_tree_model.pkl not found")

try:
    label_encoders = joblib.load(os.path.join(BASE_DIR, "label_encoders (2).pkl"))
    print("[Flask API] Label encoders loaded OK")
except FileNotFoundError:
    label_encoders = None
    print("[Flask API] WARNING: label_encoders (2).pkl not found")

try:
    df = pd.read_csv(os.path.join(BASE_DIR, "insurance_claims (1).csv"))
    print(f"[Flask API] Dataset loaded OK ({len(df)} rows)")
except FileNotFoundError:
    df = None
    print("[Flask API] WARNING: insurance_claims (1).csv not found")


# ── Encode raw user input into model-ready DataFrame ──────────────────────────
def encode_input(user_data):
    row = {}
    for feat in ALL_FEATURES:
        val = user_data.get(feat, None)
        if feat in CATEGORICAL_FEATURES:
            if label_encoders and isinstance(label_encoders, dict) and feat in label_encoders:
                le = label_encoders[feat]
                val_str = str(val) if val is not None else ""
                row[feat] = int(le.transform([val_str])[0]) if val_str in le.classes_ else 0
            else:
                try:
                    row[feat] = float(val)
                except (TypeError, ValueError):
                    row[feat] = 0
        else:
            try:
                row[feat] = float(val) if val not in (None, "", "nan") else 0.0
            except (TypeError, ValueError):
                row[feat] = 0.0
    return pd.DataFrame([row])[ALL_FEATURES].astype(float)


# ── Walk Decision Tree path to extract human-readable reasons ─────────────────
def extract_reasons(encoded_df, raw_input, prediction_label):
    if model is None:
        return []
    try:
        from sklearn.tree import _tree
        tree   = model.tree_
        sample = encoded_df.values[0]
        node, reasons, visited = 0, [], set()

        while True:
            feat_idx = tree.feature[node]
            if feat_idx == _tree.TREE_UNDEFINED:
                break
            feat_name    = ALL_FEATURES[feat_idx]
            threshold    = tree.threshold[node]
            sample_val   = sample[feat_idx]
            display_name = FEATURE_DISPLAY_NAMES.get(feat_name, feat_name)

            if feat_name not in visited:
                visited.add(feat_name)
                # Decode encoded value back to raw label for display
                raw_val = raw_input.get(feat_name, sample_val)
                if feat_name in CATEGORICAL_FEATURES and label_encoders and isinstance(label_encoders, dict):
                    le = label_encoders.get(feat_name)
                    if le is not None:
                        try:
                            raw_val = le.inverse_transform([int(sample_val)])[0]
                        except Exception:
                            raw_val = raw_input.get(feat_name, sample_val)

                # Format value
                if feat_name in MONEY_FIELDS:
                    try:
                        display_val = "${:,}".format(int(float(raw_val)))
                    except (TypeError, ValueError):
                        display_val = str(raw_val)
                elif feat_name in NUMERICAL_FEATURES:
                    try:
                        display_val = str(int(float(raw_val)))
                    except (TypeError, ValueError):
                        display_val = str(raw_val)
                else:
                    display_val = str(raw_val)

                went_left = sample_val <= threshold
                thr_str   = "${:,}".format(int(threshold)) if feat_name in MONEY_FIELDS else "{:.2f}".format(threshold)

                if prediction_label == "Fraud":
                    if feat_name == "incident_severity":
                        impact = "Severity '{}' matched the fraud classification branch in the decision tree.".format(display_val)
                    elif feat_name in MONEY_FIELDS:
                        direction = "exceeds" if not went_left else "is within"
                        impact = "Amount {} {} the decision tree threshold ({}), contributing to fraud classification.".format(display_val, direction, thr_str)
                    elif feat_name == "witnesses":
                        impact = "Low witness count ({}) directed this claim to the fraud branch (threshold {}).".format(display_val, thr_str)
                    else:
                        cond = "<= {}".format(thr_str) if went_left else "> {}".format(thr_str)
                        impact = "'{}' value {} (condition: {}) directed this claim down the fraud branch.".format(display_name, display_val, cond)
                else:
                    if feat_name in MONEY_FIELDS:
                        impact = "Amount {} is within the non-fraud range (decision threshold {}).".format(display_val, thr_str)
                    elif feat_name == "police_report_available":
                        impact = "Police report '{}' aligned with the non-fraud decision branch.".format(display_val)
                    elif feat_name == "witnesses":
                        impact = "Witness count {} matched the non-fraud branch (threshold {}).".format(display_val, thr_str)
                    else:
                        cond = "<= {}".format(thr_str) if went_left else "> {}".format(thr_str)
                        impact = "'{}' value {} (condition: {}) directed this claim down the non-fraud branch.".format(display_name, display_val, cond)

                reasons.append({"feature": display_name, "value": display_val, "impact": impact})

            node = tree.children_left[node] if sample_val <= threshold else tree.children_right[node]
            if len(reasons) >= 5:
                break

        return reasons
    except Exception as e:
        print("[Flask API] Could not extract decision path: {}".format(e))
        return []


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 1 -- GET /api/sample-claim
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/sample-claim", methods=["GET"])
def sample_claim():
    """Returns a random row from the dataset (only the 34 model features)."""
    if df is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    row    = df.sample(1).iloc[0]
    result = {}
    for feat in ALL_FEATURES:
        val = row[feat] if feat in df.columns else 0
        result[feat] = val.item() if hasattr(val, "item") else val
    return jsonify(result)


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 2 -- POST /api/predict
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accepts JSON with all 34 features, runs the Decision Tree,
    returns prediction label + explainability reasons.

    Response:
    {
        "prediction":    "Fraud" | "Non-Fraud",
        "model":         "Decision Tree",
        "reasons":       [ {"feature": "...", "value": "...", "impact": "..."}, ... ],
        "model_metrics": { "accuracy": 0.80, "precision": 0.57, "recall": 0.71, "f1": 0.64 }
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    raw_input = request.get_json(silent=True)
    if not raw_input:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    try:
        encoded_df = encode_input(raw_input)
        pred_raw   = model.predict(encoded_df)[0]

        # Map raw output to human-readable label
        if pred_raw in (1, "1", "Y", "y", True):
            prediction_label = "Fraud"
        elif pred_raw in (0, "0", "N", "n", False):
            prediction_label = "Non-Fraud"
        else:
            prediction_label = "Fraud" if str(pred_raw).strip().upper() in ("1", "Y", "FRAUD") else "Non-Fraud"

        reasons = extract_reasons(encoded_df, raw_input, prediction_label)

        return jsonify({
            "prediction":    prediction_label,
            "model":         "Decision Tree",
            "reasons":       reasons,
            "model_metrics": {
                "accuracy":  0.80,
                "precision": 0.57,
                "recall":    0.71,
                "f1":        0.64
            }
        })
    except Exception as e:
        print("[Flask API] Prediction error: {}".format(e))
        return jsonify({"error": str(e)}), 500


# ── Health check ───────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":   "ok",
        "model":    "loaded" if model is not None else "missing",
        "dataset":  "{} rows".format(len(df)) if df is not None else "missing",
        "encoders": "loaded" if label_encoders is not None else "missing"
    })


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print()
    print("=" * 55)
    print("  Insurance Fraud Detection -- Flask API")
    print("  API URL:  http://localhost:5000")
    print("  Health:   http://localhost:5000/api/health")
    print("  Then open index.html in your browser.")
    print("=" * 55)
    print()
    app.run(host="0.0.0.0", port=5000, debug=True)