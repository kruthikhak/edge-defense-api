Live demo: https://lnkd.in/gzWFwff7

# Edge Defense: Explainable AI Framework for Real-Time Encrypted Network Intrusion Detection

Edge Defense is a full-stack explainable intrusion detection system designed to classify encrypted network traffic in real time using a lightweight machine learning model. The project combines an optimized XGBoost classifier with SHAP explainability to deliver transparent attack predictions while maintaining ultra-low inference latency suitable for edge deployment.

The platform includes an end-to-end machine learning pipeline, REST API backend, interactive React dashboard, secure authentication, cloud deployment, and prediction history management.

---

## Features

- Real-time encrypted network traffic classification
- Explainable AI using SHAP for every prediction
- Lightweight edge-optimized XGBoost model
- Interactive dashboard for live traffic analysis
- Batch CSV upload and prediction
- Secure user authentication
- Prediction history and audit logging
- Production-ready REST API

---

## Tech Stack

### Machine Learning
- Python
- XGBoost
- SHAP
- Scikit-learn
- Pandas
- NumPy

### Backend
- FastAPI
- Uvicorn

### Frontend
- React (Vite)
- Tailwind CSS

### Authentication
- Clerk Authentication
- Google OAuth

### Database
- Supabase

### Deployment
- Railway (Backend)
- Vercel (Frontend)

---

## Dataset

The model is trained using the **CICIDS2017** intrusion detection dataset containing approximately **2.83 million** labeled network flows.

The preprocessing pipeline includes:

- Missing value handling
- Infinite value removal
- Data leakage prevention
- Feature scaling
- Label preservation
- Feature selection

The final model uses **20 optimized statistical flow features** selected through:

- Variance Thresholding
- Correlation Analysis
- Random Forest Feature Importance

---

## Explainable AI

Instead of treating the model as a black box, Edge Defense provides transparent predictions using SHAP.

Generated visualizations include:

- SHAP Beeswarm Plot
- Feature Importance
- Waterfall Plot
- Individual Prediction Explanations

These explanations help analysts understand why network traffic was classified as malicious.

---

## Model Performance

| Metric | Value |
|---------|-------|
| Accuracy | **99.57%** |
| ROC-AUC | **0.9998** |
| Macro F1 Score | **0.9933** |
| Inference Latency | **0.5–1.3 ms** |
| Model Size | **374 KB** |
| Attack Classes | **14** |

The system detects attacks including:

- DDoS
- PortScan
- Heartbleed
- Brute Force
- Web Attacks
- Botnet Traffic
- DoS Variants
- And other intrusion categories

without decrypting network traffic.

---

## Project Architecture

```
               Network Flow

                     │

                     ▼

           Data Preprocessing

                     │

                     ▼

           Feature Engineering

                     │

                     ▼

            XGBoost Classifier

                     │

          ┌──────────┴──────────┐

          ▼                     ▼

     Prediction            SHAP Explanation

          │                     │

          └──────────┬──────────┘

                     ▼

               FastAPI Backend

                     │

         ┌───────────┴───────────┐

         ▼                       ▼

   React Dashboard         Supabase Database

                     │

              Railway Deployment
```

---

## REST API

Example endpoints:

```
GET    /health
POST   /predict
POST   /predict-batch
POST   /explain
GET    /sample
```

---

## Frontend Features

- Live prediction interface
- Drag-and-drop CSV upload
- Prediction confidence
- SHAP explanation visualization
- Attack classification results
- Responsive dashboard
- Authentication-protected routes

---

## Authentication

Implemented using **Clerk**.

Supports:

- Email/Password Login
- Google OAuth
- Protected Routes

---

## Deployment

| Component | Platform |
|------------|----------|
| Frontend | Vercel |
| Backend | Railway |
| Database | Supabase |
| Authentication | Clerk |

---

## Folder Structure

```
EdgeDefense/
│
├── backend/
│   ├── app/
│   ├── models/
│   ├── artifacts/
│   ├── routers/
│   └── main.py
│
├── frontend/
│   ├── src/
│   ├── components/
│   ├── pages/
│   └── assets/
│
├── notebooks/
├── dataset/
├── model_training/
└── README.md
```

---

## Future Improvements

- Streaming packet inference
- Edge hardware benchmarking
- Docker deployment
- SIEM integration
- Threat intelligence enrichment

---

## Authors

Developed by:

- **Kruthikha Vishali**
- **Harishni Vasagam**

---

## License

This project is intended for educational and research purposes
