# 5G Base Station Energy Consumption Prediction using Explainable AI (XAI)

> **Master's Thesis** — Predicting 5G base station energy consumption using TabNet and ANN models with SHAP and LIME explainability.

---

## 📌 Overview

This repository contains the code and experiments for my Master's thesis on **predicting energy consumption of 5G base stations** using machine learning models enhanced with **Explainable AI (XAI)** techniques. The goal is to build accurate predictive models while maintaining interpretability — a critical requirement for sustainable 5G network management and energy optimization.

Energy consumption is one of the key operational challenges for 5G networks. By leveraging ML models and XAI, network operators can not only forecast energy usage but also **understand the driving factors**, enabling informed decisions for energy-efficient network planning.

---

## 🏗️ Project Structure

```
ms-thesis/
├── 5GEnergyConsumtion_preProcessing.ipynb              # Data preprocessing & feature engineering
├── 5G_energy_consumption_modeling_modelBuilding_v1.4.ipynb  # Model training, evaluation & XAI
├── README.md                                            # Project documentation
```

### Notebook Descriptions

| Notebook | Purpose |
|----------|---------|
| `5GEnergyConsumtion_preProcessing.ipynb` | Loads raw 5G energy consumption data, performs EDA, engineers 126 features (temporal, lag, rolling stats, interaction terms), and exports a processed CSV. |
| `5G_energy_consumption_modeling_modelBuilding_v1.4.ipynb` | Builds and trains TabNet and ANN models, evaluates performance, and applies SHAP, LIME, and TabNet's built-in attention for explainability. |

---

## 📊 Dataset

The project uses a **5G base station energy consumption dataset** containing operational metrics from multiple base stations. Key raw features include:

- **BS** — Base Station identifier
- **RUType** — Radio Unit type
- **Load** — Traffic load on the base station
- **Energy** — Energy consumption (target variable)
- **Time** — Timestamp of the measurement
- Additional configuration and operational parameters

### Feature Engineering

The preprocessing pipeline transforms raw data into **126 engineered features**, including:

| Feature Category | Examples |
|-----------------|----------|
| **Temporal** | `hour`, `day_of_week`, `day_of_month`, `month` |
| **Cyclical Encodings** | `hour_sin`, `hour_cos`, `day_of_week_sin`, `day_of_week_cos` |
| **Lag Features** | `load_lag_1` to `load_lag_24`, `Energy_lag_1` to `Energy_lag_24` |
| **Rolling Statistics** | `load_ma_3`, `load_ma_6`, `load_std_3`, `load_std_6`, etc. |
| **Interaction Terms** | `load_freq_interaction`, `load_bw_interaction`, `energy_freq_interaction` |
| **Derived Ratios** | `load_energy_ratio`, `power_efficiency` |
| **Configuration** | One-hot encoded `config_*` columns |

After one-hot encoding the `BS` column during model building, the final feature space expands to **1,045 features**.

---

## 🤖 Models

### 1. TabNet

[TabNet](https://arxiv.org/abs/1908.07442) is an attention-based deep learning model designed for tabular data. It provides built-in interpretability through its sequential attention mechanism.

- **Unsupervised pretraining** followed by supervised fine-tuning
- Architecture: `n_d=128, n_a=128, n_steps=8, gamma=1.3`
- Training: `max_epochs=200, patience=30, batch_size=256`

### 2. ANN (Artificial Neural Network)

A fully connected feedforward neural network built with PyTorch.

- Architecture: `1045 → 128 → 64 → 32 → 1` (with ReLU and Dropout)
- Optimizer: Adam (`lr=0.001`)
- Loss: Mean Squared Error (MSE)
- Training: `max_epochs=100, patience=20, batch_size=256`

---

## 📈 Results

| Model   | MSE       | MAE      | R²       |
|---------|-----------|----------|----------|
| **TabNet** | **1.3018** | **0.6568** | **0.9937** |
| ANN     | 7.9756    | 2.0614   | 0.9617   |

> **TabNet significantly outperforms the ANN** with ~6× lower MSE, ~3× lower MAE, and a higher R² score, demonstrating the effectiveness of attention-based architectures for tabular energy consumption data.

---

## Explainability (XAI)

Understanding **why** a model makes certain predictions is crucial for trust and adoption in real-world network management. This project employs three XAI techniques:

### SHAP (SHapley Additive exPlanations)
- Uses `KernelExplainer` with a background sample of 100 data points.
- Generates **global feature importance** via SHAP summary plots for both models.
- Reveals which features most influence energy consumption predictions.

### LIME (Local Interpretable Model-agnostic Explanations)
- Uses `LimeTabularExplainer` for **local (per-instance) explanations**.
- Provides intuitive, human-readable justifications for individual predictions.

### TabNet Built-in Feature Importance
- Leverages TabNet's inherent **sequential attention mechanism**.
- Extracts feature importance directly from the model's learned attention masks.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.x |
| **Deep Learning** | PyTorch, pytorch-tabnet |
| **ML & Data** | scikit-learn, pandas, NumPy |
| **XAI** | SHAP, LIME |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Jupyter Notebook |

---

## Getting Started

### Prerequisites

```bash
pip install torch pytorch-tabnet shap lime scikit-learn pandas numpy matplotlib seaborn
```

### Reproduction Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mo-arindam/ms-thesis.git
   cd ms-thesis
   ```

2. **Run preprocessing:**
   Open and execute `5GEnergyConsumtion_preProcessing.ipynb` to generate the processed dataset (`5G_energy_consumption_processed.csv`).

3. **Run model training & evaluation:**
   Open and execute `5G_energy_consumption_modeling_modelBuilding_v1.4.ipynb` to train models, evaluate performance, and generate XAI explanations.

> **Note:** The processed dataset and trained model artifacts are not included in the repository due to file size. You must run Notebook 1 before Notebook 2.

---

## Generated Artifacts (Not in Repo)

The model building notebook generates and caches the following artifacts locally:

```
artifacts/
├── tabnet_model.zip          # Trained TabNet model weights
├── ann_model.pth             # Trained ANN model weights
├── scaler_X.pkl              # Feature scaler
├── scaler_y.pkl              # Target scaler
├── feature_names.pkl         # Feature name list
└── 5G_energy_consumption_processed.csv  # Processed dataset
```

---

## License

This project is part of a Master's thesis. Please contact the author for usage permissions.

---

## Author

**Arindam Mo** — [@mo-arindam](https://github.com/mo-arindam)

---

## Acknowledgments

- [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442) by Arik & Pfister (Google Cloud AI)
- [SHAP](https://github.com/shap/shap) by Scott Lundberg
- [LIME](https://github.com/marcotcr/lime) by Marco Tulio Ribeiro