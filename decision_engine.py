"""
Decision & Recommendation Engine for SentinelOps.

Loads the trained ML, NLP, and CNN models and combines their outputs into
a single recommendation per machine (PRD Sec 4, FR-06).

Update MODEL_PATHS below to match where your saved models actually live.
"""

import pickle
import json
from datetime import datetime, timezone

MODEL_PATHS = {
    "ml_model": "models/final_xgboost_model.pkl",
    "nlp_tfidf": "models/nlp_tfidf_vectorizer.pkl",
    "nlp_issue_model": "models/nlp_issue_model.pkl",
    "nlp_component_model": "models/nlp_component_model.pkl",
    "cnn_model": "models/neu_defect_cnn.keras",
}

# Must match the exact column order the ML model was trained on
ML_FEATURE_ORDER = [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
]

FAILURE_PROBABILITY_THRESHOLD = 0.5  # tune based on your model's PR curve
CNN_CONFIDENCE_THRESHOLD = 0.6

HIGH_SEVERITY_KEYWORDS = [
    "excessive", "seizure", "beyond threshold", "beyond rated limit",
    "power failure", "sudden power loss", "overheating", "stalling",
    "deformation", "cracking", "overstrain", "abnormal current",
    "no repeatable cause",
]
MEDIUM_SEVERITY_KEYWORDS = [
    "wear", "fluctuation", "restriction", "degradation", "poor finish",
    "intermittent", "signal loss", "error code", "rising process temperature",
]
LOW_SEVERITY_KEYWORDS = [
    "minor", "routine", "no issue found", "monitored", "calibration check passed",
    "lubrication topped up", "slight noise", "scheduled inspection completed",
]
HIGH_CRITICALITY_COMPONENTS = {
    "motor", "power supply unit", "drive controller", "electrical panel",
    "inverter", "main shaft", "coupling",
}
LOW_CRITICALITY_COMPONENTS = {"gasket", "filter", "lubrication line", "belt"}


def classify_severity(issue: str, component: str) -> str:
    # keyword match on issue text takes priority over component fallback
    issue_l = issue.lower()
    for kw in HIGH_SEVERITY_KEYWORDS:
        if kw in issue_l:
            return "High"
    for kw in LOW_SEVERITY_KEYWORDS:
        if kw in issue_l:
            return "Low"
    for kw in MEDIUM_SEVERITY_KEYWORDS:
        if kw in issue_l:
            return "Medium"

    comp_l = component.lower()
    if comp_l in HIGH_CRITICALITY_COMPONENTS:
        return "High"
    if comp_l in LOW_CRITICALITY_COMPONENTS:
        return "Low"
    return "Medium"


# action lookup tables — the actual "rule engine" part
ML_ACTIONS = {
    True: "Schedule inspection within 24 hours \u2014 elevated failure risk detected.",
    False: "No immediate action \u2014 continue routine monitoring.",
}

SEVERITY_ACTIONS = {
    "High": "Dispatch technician immediately for {component} \u2014 {issue}.",
    "Medium": "Schedule maintenance for {component} within the week \u2014 {issue}.",
    "Low": "Log for routine review \u2014 {issue} on {component}.",
}

CNN_DEFECT_ACTIONS = {
    "crazing": "Inspect surface for stress-related cracking; monitor for spread.",
    "inclusion": "Flag for material quality review; may require part replacement.",
    "patches": "Inspect coating/surface treatment; possible corrosion onset.",
    "pitted_surface": "Check for corrosion or erosion; schedule surface treatment.",
    "rolled-in_scale": "Inspect for manufacturing-origin defect; assess structural impact.",
    "scratches": "Cosmetic/minor structural check; monitor, low urgency unless deep.",
}

RISK_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

_models = {}


def _load_obj(path):
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def load_models():
    """Load all saved models once. Call this at app/dashboard startup."""
    if "ml_model" not in _models:
        _models["ml_model"] = _load_obj(MODEL_PATHS["ml_model"])
    if "nlp_tfidf" not in _models:
        _models["nlp_tfidf"] = _load_obj(MODEL_PATHS["nlp_tfidf"])
    if "nlp_issue_model" not in _models:
        _models["nlp_issue_model"] = _load_obj(MODEL_PATHS["nlp_issue_model"])
    if "nlp_component_model" not in _models:
        _models["nlp_component_model"] = _load_obj(MODEL_PATHS["nlp_component_model"])

    if "cnn_model" not in _models:
        from tensorflow.keras.models import load_model
        _models["cnn_model"] = load_model(MODEL_PATHS["cnn_model"])

    return _models


def predict_ml(sensor_readings: dict) -> dict:
    """sensor_readings: dict matching ML_FEATURE_ORDER keys."""
    models = load_models()
    row = [sensor_readings[f] for f in ML_FEATURE_ORDER]
    if len(row) == 5:
        # XGBoost was trained with Type_L and Type_M one-hot encoded columns
        type_l = sensor_readings.get("Type_L", 0)
        type_m = sensor_readings.get("Type_M", 0)
        row.extend([type_l, type_m])

    proba = models["ml_model"].predict_proba([row])[0][1]
    prediction = bool(proba >= FAILURE_PROBABILITY_THRESHOLD)
    return {"failure_probability": round(float(proba), 4), "predicted_failure": prediction}


def predict_nlp(log_text: str) -> dict:
    models = load_models()
    vec = models["nlp_tfidf"].transform([log_text])
    component = models["nlp_component_model"].predict(vec)[0]
    issue = models["nlp_issue_model"].predict(vec)[0]
    severity = classify_severity(issue, component)
    return {"component": component, "issue": issue, "severity": severity}


def predict_cnn(image_array) -> dict:
    """
    image_array: shape (1, 128, 128, 3), raw pixel values in the 0-255 range.
    Don't divide by 255 first — the model's Rescaling(1./255) layer already
    handles that, so pre-normalizing will double-scale the input.
    """
    models = load_models()
    if "cnn_model" not in models:
        raise RuntimeError("CNN model not loaded \u2014 check load_models().")

    class_names = ["crazing", "inclusion", "patches", "pitted_surface",
                   "rolled-in_scale", "scratches"]
    probs = models["cnn_model"].predict(image_array)[0]
    top_idx = probs.argmax()
    return {"defect_type": class_names[top_idx], "confidence": round(float(probs[top_idx]), 4)}


def build_recommendation(machine_id: str, ml_out: dict = None,
                          nlp_out: dict = None, cnn_out: dict = None) -> dict:
    """Combines ML/NLP/CNN outputs into one recommendation. Risk level is
    the highest tier triggered by any module; actions from all triggered
    modules are kept, not just the highest-priority one."""
    triggered_by = []
    actions = []
    risk_level = "Low"

    if ml_out is not None:
        triggered_by.append("ML")
        actions.append(ML_ACTIONS[ml_out["predicted_failure"]])
        if ml_out["predicted_failure"]:
            risk_level = "High" if ml_out["failure_probability"] < 0.85 else "Critical"

    if nlp_out is not None:
        triggered_by.append("NLP")
        action = SEVERITY_ACTIONS[nlp_out["severity"]].format(
            component=nlp_out["component"], issue=nlp_out["issue"]
        )
        actions.append(action)
        if RISK_ORDER[nlp_out["severity"]] > RISK_ORDER[risk_level]:
            risk_level = nlp_out["severity"]

    if cnn_out is not None and cnn_out["confidence"] >= CNN_CONFIDENCE_THRESHOLD:
        triggered_by.append("CNN")
        defect_key = cnn_out["defect_type"].lower()
        actions.append(CNN_DEFECT_ACTIONS.get(
            defect_key, f"Review detected defect: {cnn_out['defect_type']}."
        ))
        if RISK_ORDER["Medium"] > RISK_ORDER[risk_level]:  # defect detection floors risk at Medium
            risk_level = "Medium"

    return {
        "machine_id": machine_id,
        "overall_risk_level": risk_level,
        "triggered_by": triggered_by,
        "recommended_actions": actions,
        "ml_output": ml_out,
        "nlp_output": nlp_out,
        "cnn_output": cnn_out,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    # replace with real values from your pipeline
    sensor_readings = {
        "Air temperature [K]": 300.5,
        "Process temperature [K]": 310.2,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 45.3,
        "Tool wear [min]": 120,
    }
    log_text = "Machine is showing excessive vibration and the bearing needs replacement."

    ml_out = predict_ml(sensor_readings)
    nlp_out = predict_nlp(log_text)
    # cnn_out = predict_cnn(preprocessed_image)  # uncomment once CNN is wired in

    recommendation = build_recommendation(
        machine_id="M-014", ml_out=ml_out, nlp_out=nlp_out, cnn_out=None
    )
    print(json.dumps(recommendation, indent=2))
