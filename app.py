"""
SentinelOps AI — Streamlit Frontend Dashboard
==============================================
Industrial Predictive Maintenance System Frontend.

Architecture & Modules:
  1. ML Sensor Telemetry -> predict_ml(sensor_readings)
  2. NLP Maintenance Logs -> predict_nlp(log_text)
  3. CNN Surface Inspection -> predict_cnn(image_array)
  4. Decision Engine -> build_recommendation(...)
"""

import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime

from decision_engine import (
    load_models,
    predict_ml,
    predict_nlp,
    predict_cnn,
    build_recommendation,
    ML_FEATURE_ORDER,
    CNN_CONFIDENCE_THRESHOLD,
    FAILURE_PROBABILITY_THRESHOLD,
    CNN_DEFECT_ACTIONS,
)


# ===================================================================
# Model Initialization & Caching
# ===================================================================

@st.cache_resource
def cached_load_models():
    """Cache models in memory across Streamlit interactions."""
    return load_models()


def try_load_models():
    """Ensure models load cleanly; show informative error if missing."""
    try:
        cached_load_models()
        return True
    except FileNotFoundError as e:
        st.error(f"⚠️ Model file not found: {e}. Ensure models exist in `models/`.")
        return False
    except Exception as e:
        st.error(f"⚠️ Model loading error: {e}")
        return False


# ===================================================================
# Design System (Stitch / Material 3 Tokens)
# ===================================================================

def inject_design_system():
    """Inject typography, icons, and Material 3 design tokens."""
    st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet">

    <style>
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #1a1b1f;
        }

        .material-symbols-outlined {
            font-family: 'Material Symbols Outlined' !important;
            font-size: 20px;
            vertical-align: middle;
            font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
        }
        .material-symbols-outlined.filled {
            font-variation-settings: 'FILL' 1, 'wght' 400, 'GRAD' 0, 'opsz' 24;
        }

        /* Card Container */
        .stitch-card {
            background: #ffffff;
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            border: 1px solid #e3e2e7;
            margin-bottom: 1rem;
            transition: border-color 0.2s ease;
        }
        .stitch-card:hover {
            border-color: #c1c6d7;
        }

        /* Bento KPI Card */
        .bento-kpi {
            background: #ffffff;
            border-radius: 16px;
            padding: 1.2rem 1.25rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            border: 1px solid #e3e2e7;
            min-height: 115px;
            margin-bottom: 0.75rem;
        }
        .bento-kpi-title {
            font-size: 0.8rem;
            font-weight: 600;
            color: #414755;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }
        .bento-kpi-value {
            font-size: 1.85rem;
            font-weight: 700;
            color: #1a1b1f;
            line-height: 1.1;
        }
        .bento-kpi-sub {
            font-size: 0.78rem;
            color: #717786;
            margin-top: 0.25rem;
        }

        /* Status Badges */
        .stitch-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 600;
        }
        .badge-low { background: #dcfce7; color: #166534; }
        .badge-medium { background: #fef9c3; color: #854d0e; }
        .badge-high { background: #ffdbcc; color: #7c2e00; }
        .badge-critical { background: #ffdad6; color: #93000a; }

        /* Large Hero Badge */
        .risk-badge-hero {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.6rem 1.8rem;
            border-radius: 14px;
            font-size: 1.35rem;
            font-weight: 700;
        }

        /* Tags & Pills */
        .stitch-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            background: #e0edff;
            color: #004493;
            padding: 0.3rem 0.8rem;
            border-radius: 8px;
            font-size: 0.82rem;
            font-weight: 600;
            margin-right: 0.4rem;
            margin-bottom: 0.3rem;
        }
        .stitch-tag {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            background: #f4f3f8;
            color: #1a1b1f;
            padding: 0.3rem 0.8rem;
            border-radius: 9999px;
            font-size: 0.82rem;
            font-weight: 500;
            border: 1px solid #e3e2e7;
            margin-right: 0.4rem;
            margin-bottom: 0.3rem;
        }

        /* Action Checklist Items */
        .stitch-action-item {
            background: #ffffff;
            border: 1px solid #e3e2e7;
            border-radius: 12px;
            padding: 0.85rem 1.1rem;
            margin-bottom: 0.6rem;
            font-size: 0.92rem;
            color: #1a1b1f;
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
        }
        .stitch-action-item:hover {
            background: #faf9fe;
            border-color: #0058bc;
        }

        /* Failure Banner */
        .failure-banner {
            background: #ffffff;
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            border: 1px solid #e3e2e7;
            margin-bottom: 1.25rem;
        }

        /* Data Tables */
        .stitch-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #e3e2e7;
            background: #ffffff;
            margin-bottom: 1rem;
        }
        .stitch-table th {
            background: #f4f3f8;
            padding: 0.75rem 1rem;
            text-align: left;
            font-size: 0.78rem;
            font-weight: 600;
            color: #414755;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            border-bottom: 1px solid #e3e2e7;
        }
        .stitch-table td {
            padding: 0.8rem 1rem;
            border-top: 1px solid #eeedf3;
            font-size: 0.88rem;
            color: #1a1b1f;
        }
        .stitch-table tr:hover td {
            background: #faf9fe;
        }

        /* Section Headings */
        .stitch-section-title {
            font-size: 1.15rem;
            font-weight: 600;
            color: #1a1b1f;
            margin-top: 1.5rem;
            margin-bottom: 0.85rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Demo Notice */
        .stitch-demo-badge {
            background: #fff7ed;
            border: 1px solid #ffdbcc;
            border-radius: 8px;
            padding: 0.5rem 0.85rem;
            font-size: 0.8rem;
            color: #7c2e00;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        section[data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e3e2e7;
        }

        #MainMenu, footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


# ===================================================================
# UI Helper Components
# ===================================================================

def get_badge_class(level: str) -> str:
    """Return badge color class for a given risk or severity level."""
    lvl = level.lower()
    if lvl == "low":
        return "badge-low"
    elif lvl == "medium":
        return "badge-medium"
    elif lvl == "high":
        return "badge-high"
    elif lvl == "critical":
        return "badge-critical"
    return "badge-medium"


def render_badge(level: str) -> str:
    """Render HTML status badge with an icon."""
    css_class = get_badge_class(level)
    icons = {"critical": "error", "high": "warning", "medium": "info", "low": "check_circle"}
    icon = icons.get(level.lower(), "info")
    return f'<span class="stitch-badge {css_class}"><span class="material-symbols-outlined" style="font-size:14px;">{icon}</span> {level}</span>'


def render_bento_kpi(title: str, value: str, subtext: str, icon: str, icon_color: str = "#0058bc") -> str:
    """Render a Bento KPI metric card."""
    return f"""
    <div class="bento-kpi">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
            <span class="bento-kpi-title">{title}</span>
            <span class="material-symbols-outlined" style="color:{icon_color}; font-size:22px;">{icon}</span>
        </div>
        <div class="bento-kpi-value">{value}</div>
        <div class="bento-kpi-sub">{subtext}</div>
    </div>
    """


def render_risk_implication(failure_prob: float, predicted_failure: bool) -> str:
    """Return plain-English explanation of the sensor risk implication."""
    if not predicted_failure:
        return "Machine is operating within nominal parameters. Continue routine monitoring."
    elif failure_prob >= 0.85:
        return "Critical failure risk detected — immediate technician dispatch strongly recommended."
    elif failure_prob >= FAILURE_PROBABILITY_THRESHOLD:
        return "Elevated failure risk detected — schedule physical inspection within 24 hours."
    return "Machine operating within nominal parameters."


# Curated test logs for quick demonstration
PRESET_MAINTENANCE_LOGS = [
    "Machine is showing abnormal current draw during operation.",
    "Motor is experiencing abnormal current draw and overheating under heavy load.",
    "Drive controller suffered sudden power loss during morning operational shift.",
    "Main shaft showing deformation and excessive torque beyond rated limit.",
    "Electrical panel tripped with power failure and no repeatable cause identified.",
    "Coolant pump exhibits flow restriction with rising process temperature.",
    "Inverter display shows voltage fluctuation and intermittent fault.",
    "Cutting tool has worn cutting edge leading to poor finish on workpiece.",
    "Gasket inspected during scheduled inspection completed with no issue found.",
    "Routine check on filter, lubrication topped up and calibration check passed.",
    "Minor vibration noted on belt, slight noise during operation but monitored.",
]


# ===================================================================
# Page 1: Overview
# ===================================================================

def page_overview():
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="font-size: 1.75rem; font-weight: 700; color: #1a1b1f; margin-bottom: 0.25rem;">Fleet Overview</h2>
        <p style="font-size: 0.95rem; color: #414755; margin: 0;">Real-time status monitoring and health assessment across all units.</p>
    </div>
    """, unsafe_allow_html=True)

    demo_machines = [
        {
            "machine_id": "TURB-01-A",
            "risk_level": "Low",
            "failure_prob": 0.02,
            "component": "Main Bearing",
            "issue": "Routine diagnostic passed",
            "severity": "Low",
            "recommendation": "No immediate action — continue routine monitoring.",
        },
        {
            "machine_id": "PUMP-04-B",
            "risk_level": "High",
            "failure_prob": 0.68,
            "component": "Coolant Pump",
            "issue": "Excessive vibration pattern",
            "severity": "High",
            "recommendation": "Schedule inspection within 24 hours.",
        },
        {
            "machine_id": "GEN-44-X",
            "risk_level": "Medium",
            "failure_prob": 0.12,
            "component": "Drive Controller",
            "issue": "Rising process temperature",
            "severity": "Medium",
            "recommendation": "Schedule maintenance within the week.",
        },
        {
            "machine_id": "COMP-11-Z",
            "risk_level": "Critical",
            "failure_prob": 0.87,
            "component": "Power Supply Unit",
            "issue": "Sudden power loss & torque surge",
            "severity": "High",
            "recommendation": "Dispatch technician immediately.",
        },
        {
            "machine_id": "VALV-04-Q",
            "risk_level": "Low",
            "failure_prob": 0.01,
            "component": "Hydraulic Gasket",
            "issue": "Calibration check passed",
            "severity": "Low",
            "recommendation": "Log for routine review.",
        },
    ]

    total = len(demo_machines)
    low_count = sum(1 for m in demo_machines if m["risk_level"] == "Low")
    medium_count = sum(1 for m in demo_machines if m["risk_level"] == "Medium")
    high_crit_count = sum(1 for m in demo_machines if m["risk_level"] in ("High", "Critical"))

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(render_bento_kpi("Total Fleet", str(total), "Units monitored", "precision_manufacturing", "#0058bc"), unsafe_allow_html=True)
    with col2:
        st.markdown(render_bento_kpi("Low Risk", str(low_count), "Operating normally", "check_circle", "#10b981"), unsafe_allow_html=True)
    with col3:
        st.markdown(render_bento_kpi("Medium Risk", str(medium_count), "Under watch", "info", "#f59e0b"), unsafe_allow_html=True)
    with col4:
        st.markdown(render_bento_kpi("High / Critical", str(high_crit_count), "Action required", "warning", "#ba1a1a"), unsafe_allow_html=True)
    with col5:
        st.markdown(render_bento_kpi("Active Alerts", str(high_crit_count), "Unresolved tickets", "notifications_active", "#9e3d00"), unsafe_allow_html=True)

    st.markdown('<div class="stitch-section-title"><span class="material-symbols-outlined" style="color:#0058bc;">dns</span> Fleet Machine Status</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="stitch-demo-badge"><span class="material-symbols-outlined" style="font-size:16px;">info</span> '
        'Demo data — replace with live telemetry stream when connected.</div>',
        unsafe_allow_html=True,
    )

    table_rows = "".join([
        f"""<tr>
            <td><strong>{m["machine_id"]}</strong></td>
            <td>{render_badge(m["risk_level"])}</td>
            <td><strong>{m["failure_prob"]:.0%}</strong></td>
            <td>{m["component"]}</td>
            <td>{m["issue"]}</td>
            <td>{render_badge(m["severity"])}</td>
            <td style="font-size:0.83rem; color:#414755;">{m["recommendation"]}</td>
        </tr>""" for m in demo_machines
    ])

    st.markdown(f"""
    <table class="stitch-table">
        <thead>
            <tr>
                <th>Machine ID</th>
                <th>Risk Level</th>
                <th>Failure Probability</th>
                <th>Component</th>
                <th>Issue</th>
                <th>Severity</th>
                <th>Latest Recommendation</th>
            </tr>
        </thead>
        <tbody>{table_rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<div class="stitch-section-title"><span class="material-symbols-outlined" style="color:#ba1a1a;">warning</span> Priority Alerts</div>', unsafe_allow_html=True)
    alerts = [m for m in demo_machines if m["risk_level"] in ("High", "Critical")]
    for alert in alerts:
        border_color = "#ba1a1a" if alert["risk_level"] == "Critical" else "#f59e0b"
        st.markdown(f"""
        <div class="stitch-card" style="border-left: 4px solid {border_color};">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <div style="font-weight: 700; font-size: 1rem; color: #1a1b1f; margin-bottom: 0.25rem;">
                        {alert["machine_id"]} — {alert["issue"]}
                    </div>
                    <div style="font-size: 0.85rem; color: #414755;">
                        Component: <strong>{alert["component"]}</strong> • Recommendation: {alert["recommendation"]}
                    </div>
                </div>
                <div>{render_badge(alert["risk_level"])}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ===================================================================
# Page 2: Machine Analysis
# ===================================================================

def page_machine_analysis():
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="font-size: 1.75rem; font-weight: 700; color: #1a1b1f; margin-bottom: 0.25rem;">Machine Sensor Analysis</h2>
        <p style="font-size: 0.95rem; color: #414755; margin: 0;">Evaluate operational telemetry via the Tuned XGBoost sensor failure model.</p>
    </div>
    """, unsafe_allow_html=True)

    if not try_load_models():
        return

    st.markdown('<div class="stitch-section-title"><span class="material-symbols-outlined" style="color:#0058bc;">tune</span> Operational Telemetry Inputs</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        air_temp = st.number_input("Air temperature [K]", min_value=250.0, max_value=400.0, value=300.0, step=0.1)
        rotational_speed = st.number_input("Rotational speed [rpm]", min_value=0, max_value=5000, value=1500, step=10)
        tool_wear = st.number_input("Tool wear [min]", min_value=0, max_value=500, value=100, step=1)
    with col2:
        process_temp = st.number_input("Process temperature [K]", min_value=250.0, max_value=400.0, value=310.0, step=0.1)
        torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=200.0, value=40.0, step=0.1)

    if st.button("Analyze Machine Telemetry", type="primary", use_container_width=True):
        sensor_readings = {
            ML_FEATURE_ORDER[0]: air_temp,
            ML_FEATURE_ORDER[1]: process_temp,
            ML_FEATURE_ORDER[2]: rotational_speed,
            ML_FEATURE_ORDER[3]: torque,
            ML_FEATURE_ORDER[4]: tool_wear,
        }

        with st.spinner("Running XGBoost prediction..."):
            try:
                ml_result = predict_ml(sensor_readings)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

        failure_prob = ml_result["failure_probability"]
        predicted_failure = ml_result["predicted_failure"]

        st.markdown('<div class="stitch-section-title"><span class="material-symbols-outlined" style="color:#0058bc;">analytics</span> Model Evaluation Output</div>', unsafe_allow_html=True)

        banner_theme = "border-color: #ffdad6; background: #fffcfc;" if predicted_failure else "border-color: #dcfce7; background: #fafdfa;"
        text_color = "#ba1a1a" if predicted_failure else "#166534"
        icon_name = "warning" if predicted_failure else "verified"
        status_label = "Elevated Failure Probability" if predicted_failure else "Normal Operation"
        sub_label = "Sensor parameters exceed normal operating boundaries." if predicted_failure else "All sensor parameters within nominal specifications."
        bar_fill = "#ba1a1a" if predicted_failure else "#10b981"

        st.markdown(f"""
        <div class="failure-banner" style="{banner_theme}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <div>
                    <h3 style="font-size: 1.25rem; font-weight: 700; color: {text_color}; margin: 0; display: flex; align-items: center; gap: 0.5rem;">
                        <span class="material-symbols-outlined filled" style="color: {text_color};">{icon_name}</span>
                        {status_label}
                    </h3>
                    <p style="font-size: 0.88rem; color: #414755; margin-top: 0.25rem; margin-bottom: 0;">{sub_label}</p>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 2.2rem; font-weight: 700; color: {text_color}; line-height: 1;">{failure_prob:.1%}</span>
                    <span style="display: block; font-size: 0.75rem; color: {text_color}; font-weight: 600; text-transform: uppercase;">Failure Likelihood</span>
                </div>
            </div>
            <div style="width: 100%; height: 10px; background: #e3e2e7; border-radius: 9999px; overflow: hidden;">
                <div style="width: {failure_prob * 100:.1f}%; height: 100%; background: {bar_fill}; border-radius: 9999px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="stitch-card">
                <div style="font-size: 0.8rem; font-weight: 600; color: #414755; text-transform: uppercase; margin-bottom: 0.35rem;">Binary Classification</div>
                <div style="margin-top: 0.4rem;">{render_badge("Critical" if predicted_failure else "Low")}</div>
                <div style="font-size: 0.85rem; color: #414755; margin-top: 0.5rem;">Threshold: <strong>{FAILURE_PROBABILITY_THRESHOLD:.2f}</strong></div>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            implication = render_risk_implication(failure_prob, predicted_failure)
            st.markdown(f"""
            <div class="stitch-card" style="border-left: 4px solid #0058bc;">
                <div style="font-size: 0.8rem; font-weight: 600; color: #414755; text-transform: uppercase; margin-bottom: 0.35rem;">Operational Implication</div>
                <div style="font-size: 0.9rem; color: #1a1b1f; margin-top: 0.25rem;">{implication}</div>
            </div>
            """, unsafe_allow_html=True)


# ===================================================================
# Page 3: Log Analysis
# ===================================================================

def page_log_analysis():
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="font-size: 1.75rem; font-weight: 700; color: #1a1b1f; margin-bottom: 0.25rem;">Maintenance Log Analysis</h2>
        <p style="font-size: 0.95rem; color: #414755; margin: 0;">NLP-based entity extraction and rule-based severity categorization.</p>
    </div>
    """, unsafe_allow_html=True)

    if not try_load_models():
        return

    st.markdown('<div class="stitch-section-title"><span class="material-symbols-outlined" style="color:#0058bc;">edit_note</span> Unstructured Maintenance Log Input</div>', unsafe_allow_html=True)

    selected_sample = st.selectbox("Quick-fill with sample log:", ["Custom Input..."] + PRESET_MAINTENANCE_LOGS)
    default_val = selected_sample if selected_sample != "Custom Input..." else "Machine is showing abnormal current draw during operation."

    log_text = st.text_area("Log Text", value=default_val, height=110, label_visibility="collapsed")

    if st.button("Extract Entities & Analyze Severity", type="primary", use_container_width=True):
        if not log_text.strip():
            st.warning("Please enter a maintenance log entry.")
            return

        with st.spinner("Extracting entities..."):
            try:
                nlp_result = predict_nlp(log_text)
            except Exception as e:
                st.error(f"NLP error: {e}")
                return

        st.markdown('<div class="stitch-section-title"><span class="material-symbols-outlined" style="color:#0058bc;">fact_check</span> Extracted Entities</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="stitch-card">
                <div style="font-size: 0.8rem; font-weight: 600; color: #414755; text-transform: uppercase;">Component</div>
                <div style="margin-top: 0.4rem;">
                    <span class="stitch-tag"><span class="material-symbols-outlined" style="font-size:16px; color:#0058bc;">settings</span> {nlp_result["component"]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stitch-card">
                <div style="font-size: 0.8rem; font-weight: 600; color: #414755; text-transform: uppercase;">Issue</div>
                <div style="margin-top: 0.4rem;">
                    <span class="stitch-tag"><span class="material-symbols-outlined" style="font-size:16px; color:#ba1a1a;">build_circle</span> {nlp_result["issue"]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="stitch-card">
                <div style="font-size: 0.8rem; font-weight: 600; color: #414755; text-transform: uppercase;">Severity</div>
                <div style="margin-top: 0.4rem;">{render_badge(nlp_result["severity"])}</div>
                <div style="font-size: 0.75rem; color: #717786; margin-top: 0.4rem; font-style: italic;">Rule engine — not model confidence</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stitch-card" style="margin-top: 0.5rem;">
            <div style="font-size: 0.8rem; font-weight: 600; color: #414755; text-transform: uppercase; margin-bottom: 0.35rem;">Log Summary</div>
            <p style="font-size: 0.92rem; color: #1a1b1f; margin-bottom: 0.5rem;">"{log_text}"</p>
            <div style="display: flex; gap: 0.4rem; align-items: center;">
                {render_badge(nlp_result["severity"])}
                <span class="stitch-tag">{nlp_result["component"]}</span>
                <span class="stitch-tag">{nlp_result["issue"]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ===================================================================
# Page 4: Visual Inspection
# ===================================================================

def page_visual_inspection():
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="font-size: 1.75rem; font-weight: 700; color: #1a1b1f; margin-bottom: 0.25rem;">Visual Surface Inspection</h2>
        <p style="font-size: 0.95rem; color: #414755; margin: 0;">Deep learning CNN classifier for surface defect identification (NEU dataset).</p>
    </div>
    """, unsafe_allow_html=True)

    if not try_load_models():
        return

    st.markdown(
        f'<div class="stitch-demo-badge"><span class="material-symbols-outlined" style="font-size:16px;">verified</span> '
        f'Classes: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches. '
        f'Threshold: <strong>{CNN_CONFIDENCE_THRESHOLD:.0%}</strong>.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload surface image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col_img, col_info = st.columns([1, 1])
        with col_img:
            st.image(image, caption="Uploaded Surface Sample", use_container_width=True)

        with col_info:
            if st.button("Run Visual Inspection", type="primary", use_container_width=True):
                with st.spinner("Classifying with CNN..."):
                    try:
                        image_resized = image.resize((128, 128))
                        image_array = np.array(image_resized.convert("RGB"))
                        image_array = np.expand_dims(image_array, axis=0)
                        cnn_result = predict_cnn(image_array)
                    except Exception as e:
                        st.error(f"CNN error: {e}")
                        return

                defect_type = cnn_result["defect_type"]
                confidence = cnn_result["confidence"]
                above_thresh = confidence >= CNN_CONFIDENCE_THRESHOLD

                st.markdown(f"""
                <div class="stitch-card">
                    <div style="font-size: 0.8rem; font-weight: 600; color: #414755; text-transform: uppercase;">Predicted Defect</div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: #1a1b1f; margin-top: 0.3rem;">
                        {defect_type.replace('_', ' ').title()}
                    </div>
                    <div style="margin-top: 1rem;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 0.25rem;">
                            <span style="color: #414755; font-weight: 600;">Model Confidence</span>
                            <span style="color: {'#10b981' if above_thresh else '#717786'}; font-weight: 700;">{confidence:.1%}</span>
                        </div>
                        <div style="width: 100%; height: 8px; background: #e3e2e7; border-radius: 9999px; overflow: hidden;">
                            <div style="width: {confidence * 100:.1f}%; height: 100%; background: {'#10b981' if above_thresh else '#f59e0b'}; border-radius: 9999px;"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if above_thresh:
                    defect_action = CNN_DEFECT_ACTIONS.get(defect_type.lower(), f"Review detected defect: {defect_type}.")
                    st.markdown(f"""
                    <div class="stitch-card" style="border-left: 4px solid #f59e0b;">
                        <div style="font-size: 0.8rem; font-weight: 600; color: #414755; text-transform: uppercase;">Prescribed Action</div>
                        <div style="font-size: 0.92rem; color: #1a1b1f; margin-top: 0.25rem;">{defect_action}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info(f"Confidence ({confidence:.1%}) is below threshold ({CNN_CONFIDENCE_THRESHOLD:.0%}). No action generated.")


# ===================================================================
# Page 5: Recommendations
# ===================================================================

def page_recommendations():
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="font-size: 1.75rem; font-weight: 700; color: #1a1b1f; margin-bottom: 0.25rem;">Unified Maintenance Recommendations</h2>
        <p style="font-size: 0.95rem; color: #414755; margin: 0;">Multi-modal intelligence combining Sensor ML, Shift Log NLP, and Surface CNN.</p>
    </div>
    """, unsafe_allow_html=True)

    if not try_load_models():
        return

    with st.expander("Case Input Configuration", expanded=True):
        machine_id = st.text_input("Machine ID", value="M-014")

        st.markdown("**1. Sensor Telemetry Readings**")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            air_temp = st.number_input("Air temp [K]", value=300.5, step=0.1, key="r_air")
            rot_speed = st.number_input("Speed [rpm]", value=1500, step=10, key="r_spd")
        with sc2:
            proc_temp = st.number_input("Process temp [K]", value=310.2, step=0.1, key="r_prc")
            torque = st.number_input("Torque [Nm]", value=45.3, step=0.1, key="r_trq")
        with sc3:
            tool_wear = st.number_input("Tool wear [min]", value=120, step=1, key="r_tlw")

        st.markdown("**2. Maintenance Shift Log**")
        selected_rec_log = st.selectbox("Quick-fill shift log:", ["Custom Input..."] + PRESET_MAINTENANCE_LOGS, key="r_pick")
        default_rec_val = selected_rec_log if selected_rec_log != "Custom Input..." else "Machine is showing excessive vibration and the bearing needs replacement."
        log_text = st.text_area("Shift Log Text", value=default_rec_val, height=80, key="r_log", label_visibility="collapsed")

        st.markdown("**3. Optional Surface Inspection Image**")
        rec_image_file = st.file_uploader("Surface image", type=["png", "jpg", "jpeg"], key="r_img")

    if st.button("Generate Unified Recommendation", type="primary", use_container_width=True):
        with st.spinner("Synthesizing multi-modal telemetry..."):
            try:
                sensor_data = {
                    ML_FEATURE_ORDER[0]: air_temp,
                    ML_FEATURE_ORDER[1]: proc_temp,
                    ML_FEATURE_ORDER[2]: rot_speed,
                    ML_FEATURE_ORDER[3]: torque,
                    ML_FEATURE_ORDER[4]: tool_wear,
                }
                ml_out = predict_ml(sensor_data)
                nlp_out = predict_nlp(log_text) if log_text.strip() else None

                cnn_out = None
                if rec_image_file is not None:
                    img = Image.open(rec_image_file)
                    img_res = img.resize((128, 128))
                    img_arr = np.array(img_res.convert("RGB"))
                    img_arr = np.expand_dims(img_arr, axis=0)
                    cnn_out = predict_cnn(img_arr)

                rec = build_recommendation(
                    machine_id=machine_id,
                    ml_out=ml_out,
                    nlp_out=nlp_out,
                    cnn_out=cnn_out,
                )
            except Exception as e:
                st.error(f"Decision Engine error: {e}")
                return

        st.markdown('<div class="stitch-section-title"><span class="material-symbols-outlined" style="color:#0058bc;">assignment_turned_in</span> Decision Engine Assessment</div>', unsafe_allow_html=True)

        risk_lvl = rec["overall_risk_level"]
        st.markdown(f"""
        <div class="stitch-card" style="text-align: center; padding: 1.5rem;">
            <div style="font-size: 0.8rem; font-weight: 600; color: #414755; text-transform: uppercase; margin-bottom: 0.5rem;">
                Overall Machine Risk Assessment
            </div>
            <div>
                <span class="risk-badge-hero {get_badge_class(risk_lvl)}">
                    <span class="material-symbols-outlined filled" style="font-size:24px;">security</span>
                    {risk_lvl} Risk
                </span>
            </div>
            <div style="font-size: 0.85rem; color: #717786; margin-top: 0.75rem;">
                Machine: <strong>{rec["machine_id"]}</strong> • Timestamp: {rec["timestamp"][:19]} UTC
            </div>
        </div>
        """, unsafe_allow_html=True)

        triggers_html = "".join([f'<span class="stitch-pill"><span class="material-symbols-outlined" style="font-size:16px;">hub</span> {t} Module</span>' for t in rec["triggered_by"]])
        st.markdown(f"""
        <div class="stitch-card">
            <div style="font-size: 0.8rem; font-weight: 600; color: #414755; text-transform: uppercase; margin-bottom: 0.5rem;">Contributing Intelligence Modules</div>
            <div>{triggers_html}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="stitch-section-title"><span class="material-symbols-outlined" style="color:#0058bc;">checklist</span> Prescribed Action Items</div>', unsafe_allow_html=True)
        for idx, act in enumerate(rec["recommended_actions"], 1):
            st.markdown(f"""
            <div class="stitch-action-item">
                <span class="material-symbols-outlined" style="color:#0058bc; font-size:20px; margin-top:1px;">check_box_outline_blank</span>
                <div style="flex-grow: 1;">
                    <div style="font-weight: 600; color: #1a1b1f;">Action {idx}</div>
                    <div style="font-size: 0.9rem; color: #414755; margin-top: 0.2rem;">{act}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="stitch-section-title"><span class="material-symbols-outlined" style="color:#0058bc;">code</span> Viva Explainability: Raw Module Payloads</div>', unsafe_allow_html=True)
        with st.expander("Machine Learning Payload (predict_ml)"):
            st.json(rec["ml_output"])
        with st.expander("Natural Language Processing Payload (predict_nlp)"):
            st.json(rec["nlp_output"])
        with st.expander("Computer Vision Payload (predict_cnn)"):
            if rec["cnn_output"]:
                st.json(rec["cnn_output"])
            else:
                st.write("No image provided for CNN evaluation.")


# ===================================================================
# Page 6: Model Performance
# ===================================================================

def page_model_performance():
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="font-size: 1.75rem; font-weight: 700; color: #1a1b1f; margin-bottom: 0.25rem;">Model Performance & Validation Metrics</h2>
        <p style="font-size: 0.95rem; color: #414755; margin: 0;">Empirical validation metrics sourced directly from training notebooks.</p>
    </div>
    """, unsafe_allow_html=True)

    # ML XGBoost
    st.markdown('<div class="stitch-section-title"><span class="material-symbols-outlined" style="color:#0058bc;">memory</span> ML Sensor Model — Tuned XGBoost</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stitch-card">
        <p style="font-size: 0.88rem; color: #414755; margin-bottom: 0.75rem;">
            Trained on AI4I 2020 Predictive Maintenance dataset (10,000 samples, 3.39% imbalanced failure rate). Selected for highest test F1-score.
        </p>
        <table class="stitch-table">
            <thead>
                <tr><th>Metric</th><th>Score</th></tr>
            </thead>
            <tbody>
                <tr><td>Accuracy</td><td><strong>98.55%</strong></td></tr>
                <tr><td>Precision</td><td><strong>83.33%</strong></td></tr>
                <tr><td>Recall</td><td><strong>65.57%</strong></td></tr>
                <tr><td>F1-Score</td><td><strong>73.39%</strong></td></tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("XGBoost Confusion Matrix"):
        st.markdown("""
        <table class="stitch-table">
            <thead>
                <tr>
                    <th>Actual \\ Predicted</th>
                    <th>Pred. No Failure</th>
                    <th>Pred. Failure</th>
                </tr>
            </thead>
            <tbody>
                <tr><td><strong>Actual No Failure</strong></td><td>1931 (TN)</td><td>8 (FP)</td></tr>
                <tr><td><strong>Actual Failure</strong></td><td>21 (FN)</td><td>40 (TP)</td></tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

    # NLP Models
    st.markdown('<div class="stitch-section-title"><span class="material-symbols-outlined" style="color:#0058bc;">description</span> NLP Module — Component & Issue Classifiers</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stitch-card">
        <p style="font-size: 0.88rem; color: #414755; margin-bottom: 0.75rem;">
            TF-IDF Vectorizer + Logistic Regression models trained on maintenance log corpus. Severity resolved via keyword rule engine.
        </p>
        <table class="stitch-table">
            <thead>
                <tr><th>Task</th><th>Accuracy</th><th>Weighted Precision</th><th>Weighted Recall</th><th>Weighted F1</th></tr>
            </thead>
            <tbody>
                <tr><td>Component Extraction</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr>
                <tr><td>Issue Extraction</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr>
            </tbody>
        </table>
        <div style="font-size: 0.78rem; color: #717786; font-style: italic;">
            Note: 100% test metrics reflect the synthetic nature of the curated maintenance log training dataset.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # CNN Defect Model
    st.markdown('<div class="stitch-section-title"><span class="material-symbols-outlined" style="color:#0058bc;">photo_camera</span> CNN Visual Inspection — Surface Defect Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stitch-card">
        <p style="font-size: 0.88rem; color: #414755; margin-bottom: 0.75rem;">
            Custom Sequential CNN (Conv2D + MaxPool + Rescaling) trained on NEU Surface Defect Database across 6 defect classes.
        </p>
        <table class="stitch-table">
            <thead>
                <tr><th>Validation Metric</th><th>Value</th></tr>
            </thead>
            <tbody>
                <tr><td>Validation Accuracy</td><td><strong>87.78%</strong></td></tr>
                <tr><td>Validation Loss</td><td><strong>0.2531</strong></td></tr>
                <tr><td>Macro F1-Score</td><td><strong>0.88</strong></td></tr>
                <tr><td>Weighted F1-Score</td><td><strong>0.88</strong></td></tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("CNN Per-Class Performance"):
        st.markdown("""
        <table class="stitch-table">
            <thead>
                <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>
            </thead>
            <tbody>
                <tr><td>Crazing</td><td>0.91</td><td>1.00</td><td>0.95</td><td>60</td></tr>
                <tr><td>Inclusion</td><td>0.91</td><td>0.68</td><td>0.78</td><td>60</td></tr>
                <tr><td>Patches</td><td>1.00</td><td>1.00</td><td>1.00</td><td>60</td></tr>
                <tr><td>Pitted Surface</td><td>0.74</td><td>0.87</td><td>0.80</td><td>60</td></tr>
                <tr><td>Rolled-in Scale</td><td>1.00</td><td>0.82</td><td>0.90</td><td>60</td></tr>
                <tr><td>Scratches</td><td>0.77</td><td>0.90</td><td>0.83</td><td>60</td></tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)


# ===================================================================
# Main Router
# ===================================================================

def main():
    st.set_page_config(
        page_title="SentinelOps AI",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_design_system()

    with st.sidebar:
        st.markdown("""
        <div style="padding: 0.5rem 0 1rem 0;">
            <div style="display: flex; align-items: center; gap: 0.6rem;">
                <div style="width: 38px; height: 38px; border-radius: 10px; background: #0058bc; display: flex; align-items: center; justify-content: center; color: white;">
                    <span class="material-symbols-outlined filled" style="font-size: 22px;">shield</span>
                </div>
                <div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #1a1b1f; line-height: 1.1;">SentinelOps</div>
                    <div style="font-size: 0.75rem; color: #717786;">Industrial AI Dashboard</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        page = st.radio(
            "Navigation",
            options=[
                "Overview",
                "Machine Analysis",
                "Log Analysis",
                "Visual Inspection",
                "Recommendations",
                "Model Performance",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.75rem; color: #717786; line-height: 1.5;">
            <strong>System:</strong> Operational<br>
            <strong>Version:</strong> v1.0 (Final)<br>
            <strong>Decision Engine:</strong> Integrated
        </div>
        """, unsafe_allow_html=True)

    if page == "Overview":
        page_overview()
    elif page == "Machine Analysis":
        page_machine_analysis()
    elif page == "Log Analysis":
        page_log_analysis()
    elif page == "Visual Inspection":
        page_visual_inspection()
    elif page == "Recommendations":
        page_recommendations()
    elif page == "Model Performance":
        page_model_performance()


if __name__ == "__main__":
    main()
