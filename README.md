# SentinelOps AI

An AI-powered industrial predictive maintenance system designed to monitor machine health, analyze maintenance logs, classify surface defects from equipment imagery, and generate prioritized, actionable maintenance recommendations.

---

## 🏗️ System Architecture

```text
                     ┌────────────────────────┐
                     │  Sensor Telemetry Data │
                     └───────────┬────────────┘
                                 ↓
                     ┌────────────────────────┐
                     │   ML Prediction Model  │ (Tuned XGBoost)
                     └───────────┬────────────┘
                                 ↓
                        Failure Probability
                                 │
 ┌──────────────────────┐        │        ┌────────────────────────┐
 │   Maintenance Logs   │        │        │ Equipment Surface Img  │
 └──────────┬───────────┘        │        └───────────┬────────────┘
            ↓                    │                    ↓
 ┌──────────────────────┐        │        ┌────────────────────────┐
 │   NLP Entity Model   │        │        │   CNN Defect Model     │
 └──────────┬───────────┘        │        └───────────┬────────────┘
            │ (Component, Issue, │                    │ (Defect Type,
            │  Rule Severity)    │                    │  Confidence)
            └───────────┐        │        ┌───────────┘
                        ↓        ↓        ↓
                 ┌────────────────────────────────┐
                 │  Decision Engine Integration   │
                 │    (build_recommendation)      │
                 └───────────────┬────────────────┘
                                 ↓
                 ┌────────────────────────────────┐
                 │  Streamlit Frontend Dashboard  │
                 └────────────────────────────────┘
```

---

## 📦 Core Intelligence Modules

### 1. Machine Learning — Sensor Failure Prediction
- **Dataset:** AI4I 2020 Predictive Maintenance Dataset (10,000 observations, 3.39% class imbalance).
- **Features:** Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min], Machine Type.
- **Selected Model:** **Tuned XGBoost** (`final_xgboost_model.pkl`).
- **Performance:**
  - **Accuracy:** 98.55%
  - **Precision:** 83.33%
  - **Recall:** 65.57% (detected 40 of 61 actual test failures)
  - **F1-Score:** **73.39%** (highest across all evaluated algorithms)
- **Confusion Matrix:** True Negatives: 1931 | False Positives: 8 | False Negatives: 21 | True Positives: 40.

### 2. Natural Language Processing — Log Analysis
- **Pipeline:** TF-IDF Vectorization + Logistic Regression.
- **Models:**
  - `nlp_component_model.pkl`: Classifies component (motor, pump, bearing, valve, etc.).
  - `nlp_issue_model.pkl`: Classifies mechanical/electrical issue (abnormal current draw, rising temp, vibration, etc.).
- **Severity Rule Engine:** Implemented via `classify_severity()` in `decision_engine.py` using keyword hierarchy and component criticality policies rather than uncalibrated confidence scores.

### 3. Computer Vision — Surface Defect Inspection
- **Dataset:** NEU Surface Defect Database.
- **Architecture:** Custom Sequential CNN (Conv2D + MaxPooling + Dense) with built-in `Rescaling(1./255)`.
- **Target Classes (6):** `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`.
- **Validation Accuracy:** **87.78%** | **Validation Loss:** **0.2531** | **Macro F1:** **0.88**.
- **Action Policy:** Only generates defect maintenance actions when model confidence meets or exceeds `CNN_CONFIDENCE_THRESHOLD = 0.60`.

### 4. Decision & Recommendation Engine
- **Module:** `decision_engine.py`
- **Core Function:** `build_recommendation(machine_id, ml_out, nlp_out, cnn_out)`
- **Key Logic:**
  - Aggregates multi-modal intelligence into an overall risk tier (`Low`, `Medium`, `High`, `Critical`).
  - Tracks which AI modules contributed (`triggered_by`).
  - Generates a full checklist of prescribed operational actions without discarding lower-priority alerts.

---

## 💻 Streamlit Frontend Dashboard

The frontend (`app.py`) is styled using the **Stitch / Material 3 Design System** (`Inter` typography, `#0058BC` primary palette, `#FAF9FE` canvas, `16px` rounded cards, and Google Material Symbols Outlined).

### Dashboard Pages:
1. **Fleet Overview:** KPI metrics (Total Units, Risk Counts, Active Alerts), interactive status table with demo data, and priority alert feeds.
2. **Machine Analysis:** Telemetry input form connected live to `predict_ml()`, displaying failure probability bar and risk implications.
3. **Log Analysis:** Unstructured log text analyzer with quick-fill presets, extracting components, issues, and rule-based severity.
4. **Visual Inspection:** Image uploader connected to `predict_cnn()`, featuring confidence meter and defect action prescriptions.
5. **Unified Recommendations:** Multi-modal case generator synthesizing ML + NLP + CNN inputs into unified decision reports with action checklists.
6. **Model Performance:** Comprehensive validation benchmark reports, per-class metrics, and confusion matrices.

---

## 📁 Repository Structure

```text
SentinelOps/
├── .streamlit/
│   └── config.toml          # Light theme config (Primary #0058BC, Canvas #FAF9FE)
├── data/                    # Processed datasets (AI4I 2020 & Maintenance logs)
│   ├── ai4i2020.csv
│   ├── feature_engineered_data.csv
│   └── sentinelops_nlp_maintenance_logs.csv
├── models/                  # Trained serialized model artifacts
│   ├── final_xgboost_model.pkl
│   ├── neu_defect_cnn.keras
│   ├── nlp_component_model.pkl
│   ├── nlp_issue_model.pkl
│   └── nlp_tfidf_vectorizer.pkl
├── Notebooks/               # Research & training notebooks
│   ├── sentinelops-EDA.ipynb
│   ├── sentinelops-feature-engineering.ipynb
│   ├── sentinelops-model-training.ipynb
│   ├── setinelops-nlp.ipynb
│   ├── sentinelops-cnn.ipynb
│   └── sentinelops-prediction.ipynb
├── app.py                   # Streamlit frontend dashboard (6 pages)
├── decision_engine.py       # Decision & Recommendation Engine
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 recommended.

### 2. Installation
```bash
# Clone repository
git clone https://github.com/geekyfromgreek/SentinelOps_final_project.git
cd SentinelOps

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Launch Dashboard
```bash
streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.

---

## ✅ Current Project Status

- [x] Exploratory Data Analysis & Preprocessing
- [x] Feature Engineering & Leakage Removal
- [x] Multi-Model Benchmarking & Hyperparameter Tuning
- [x] Final ML Model Export (Tuned XGBoost)
- [x] NLP Log Analysis Pipeline (TF-IDF + Logistic Regression)
- [x] Rule-Based Severity Classification Engine
- [x] CNN Visual Surface Defect Classifier (NEU Dataset)
- [x] Decision & Recommendation Engine Integration
- [x] Streamlit Frontend Dashboard (6 Multi-Modal Pages)
- [x] Material 3 / Stitch Design System Implementation
- [x] Full System Testing & Verification

---

## 🔮 Future Scope & Production Roadmap

To scale SentinelOps AI from an MVP prototype into an **end-to-end enterprise production system**, the following architectural and data improvements are planned:

### 1. Advanced Data Acquisition & Quality
- **High-Frequency Sensor Streams:** Ingest continuous high-frequency time-series telemetry (vibration acceleration, acoustic emissions, and motor current signature analysis) rather than aggregated tabular summaries.
- **Real-World Maintenance Logs:** Expand beyond synthetic log datasets by integrating authentic ERP/CMMS shift logs (e.g., SAP PM, IBM Maximo) to train deep NLP transformers (BERT / RoBERTa / LLM-based extractors).
- **Multi-Modal Visual Data:** Collect high-resolution, multi-angle industrial surface imagery across varied lighting conditions and diverse alloy/metal materials.

### 2. High-Performance FastAPI Microservices
- **Decoupled API Backend:** Transition model inference logic from the frontend runtime into asynchronous **FastAPI** REST / gRPC microservices:
  - `/api/v1/predict/sensor` — High-throughput streaming sensor evaluation.
  - `/api/v1/predict/log` — Asynchronous NLP text entity and issue extraction.
  - `/api/v1/predict/visual` — GPU-accelerated batch image defect classification.
  - `/api/v1/recommendation` — Unified decision engine orchestration endpoint.
- **Message Queues & Streaming:** Implement **Apache Kafka** or **RabbitMQ** with **Celery/Redis** workers to process massive IoT telemetry streams asynchronously without blocking.

### 3. Containerization & Kubernetes Orchestration (Docker & K8s)
- **Dockerization:** Multi-stage Docker container builds for lightweight, reproducible, and secure container images across the API gateway, model inference services, and frontend dashboard.
- **Kubernetes (K8s) Cluster Deployment:**
  - **Horizontal Pod Autoscaling (HPA):** Dynamically scale inference pods up or down based on CPU/GPU utilization and telemetry ingestion traffic.
  - **Zero-Downtime Rolling Updates:** Deploy updated model weights and features seamlessly without service interruption.
  - **Resilience & Health Probing:** Automated liveness and readiness probes ensuring self-healing and high availability (99.99% uptime).

### 4. End-to-End MLOps Pipeline
- Automated model retraining, data versioning (DVC), experiment tracking (**MLflow** / **Weights & Biases**), and model drift detection for continuous deployment and performance monitoring.

