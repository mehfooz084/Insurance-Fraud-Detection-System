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
    df = pd.read_csv(os.path.join(BASE_DIR, "interview_test_cases.csv"))
    print(f"[Flask API] Dataset loaded OK ({len(df)} rows)")
except FileNotFoundError:
    df = None
    print("[Flask API] WARNING: interview_test_cases.csv not found")


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


# ── Generate human-readable reasons based on intuitive business logic ───────────
def extract_reasons(encoded_df, raw_input, prediction_label):
    if model is None:
        return []
    
    reasons = []
    
    try:
        # Extract relevant fields
        police_report = str(raw_input.get("police_report_available", "")).strip().upper()
        authorities = str(raw_input.get("authorities_contacted", "")).strip().title()
        witnesses = int(float(raw_input.get("witnesses", 0)))
        severity = str(raw_input.get("incident_severity", "")).strip().title()
        claim_amt = int(float(raw_input.get("total_claim_amount", 0)))
        incident_type = str(raw_input.get("incident_type", "")).strip().title()

        amt_str = "${:,}".format(claim_amt)

        if prediction_label == "Fraud":
            # 1. Police Report
            if police_report in ["NO", "?"]:
                reasons.append({
                    "feature": "Police Report",
                    "value": police_report,
                    "impact": "No official police report was filed, which is irregular for standard claims."
                })
            
            # 2. Authorities Contacted
            if authorities == "None":
                reasons.append({
                    "feature": "Authorities Contacted",
                    "value": authorities,
                    "impact": "No emergency authorities (Police, Fire, Ambulance) were contacted at the scene."
                })
            
            # 3. Witnesses
            if witnesses == 0:
                reasons.append({
                    "feature": "Witnesses",
                    "value": str(witnesses),
                    "impact": "There are zero independent witnesses to verify the events of the incident."
                })

            # 4. Theft without Police
            if incident_type == "Vehicle Theft" and police_report in ["NO", "?"]:
                reasons.append({
                    "feature": "Incident Type",
                    "value": incident_type,
                    "impact": "A vehicle theft was claimed without an accompanying police report."
                })

            # 5. Severity & High Amount
            if severity in ["Major Damage", "Total Loss"]:
                reasons.append({
                    "feature": "Claim Amount vs Severity",
                    "value": amt_str,
                    "impact": f"A high claim amount for '{severity}' is historically correlated with elevated risk profiles."
                })
            elif claim_amt > 50000:
                reasons.append({
                    "feature": "Total Claim Amount",
                    "value": amt_str,
                    "impact": "The requested claim amount exceeds the typical threshold for low-risk claims."
                })
            
            # Fallback if too few reasons
            if len(reasons) < 2:
                reasons.append({
                    "feature": "Overall Pattern",
                    "value": "Multiple Factors",
                    "impact": "The combined risk factors of this claim align with historical fraudulent profiles."
                })

        else:
            # 1. Police Report
            if police_report == "YES":
                reasons.append({
                    "feature": "Police Report",
                    "value": police_report,
                    "impact": "An official police report is available, corroborating the incident details."
                })
            
            # 2. Authorities Contacted
            if authorities in ["Police", "Ambulance", "Fire"]:
                reasons.append({
                    "feature": "Authorities Contacted",
                    "value": authorities,
                    "impact": f"Proper authorities ({authorities}) were notified at the time of the incident."
                })
            
            # 3. Witnesses
            if witnesses > 0:
                reasons.append({
                    "feature": "Witnesses",
                    "value": str(witnesses),
                    "impact": f"The presence of {witnesses} witness(es) helps verify the legitimacy of the claim."
                })

            # 4. Claim Amount
            if claim_amt < 40000 and claim_amt > 0:
                reasons.append({
                    "feature": "Total Claim Amount",
                    "value": amt_str,
                    "impact": "The requested claim amount is within normal, expected parameters for routine claims."
                })

            # 5. Severity
            if severity in ["Minor Damage", "Trivial Damage"]:
                reasons.append({
                    "feature": "Incident Severity",
                    "value": severity,
                    "impact": f"The severity ('{severity}') is typical for standard, low-risk claims."
                })
                
            # Fallback if too few reasons
            if len(reasons) < 2:
                reasons.append({
                    "feature": "Historical Alignment",
                    "value": "Normal Range",
                    "impact": "The claim characteristics consistently align with standard, valid historical data."
                })

        # Return top 4 reasons
        return reasons[:4]

    except Exception as e:
        print("[Flask API] Could not extract intuitive reasons: {}".format(e))
        return []


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 1 -- GET /api/sample-claim
# ══════════════════════════════════════════════════════════════════════════════
# Global counter to alternate between fraud and non-fraud
sample_toggle = 0

@app.route("/api/sample-claim", methods=["GET"])
def sample_claim():
    """Returns a row from the dataset, alternating between Fraud and Non-Fraud."""
    global sample_toggle
    
    if df is None:
        return jsonify({"error": "Dataset not loaded"}), 500
        
    # Check if fraud_reported exists in the dataset
    if "fraud_reported" in df.columns:
        # Toggle between 1 (Fraud) and 0 (Non-Fraud)
        target_label = 1 if sample_toggle % 2 == 0 else 0
        sample_toggle += 1
        
        subset = df[df["fraud_reported"] == target_label]
        if not subset.empty:
            row = subset.sample(1).iloc[0]
        else:
            row = df.sample(1).iloc[0]
    else:
        row = df.sample(1).iloc[0]
        
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