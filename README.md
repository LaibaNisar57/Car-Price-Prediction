# Car-Price-Prediction
## 👤 Author
- **Program:** EXPS Nexus Virtual Internship
- **Role:** Data Science Intern
- **Name:** Laiba Nisar

# Used Vehicle Valuation Engine: Car Price Prediction with Machine Learning

An end-to-end econometric and machine learning pipeline designed to clean real-world vehicle transaction records, engineer domain-specific depreciation features, and predict market prices using regularized and tree-based regression models.

## 📌 Project Overview
Used vehicle pricing depends heavily on non-linear depreciation patterns, brand prestige, and physical condition. This project sanitizes a noisy dataset containing multi-unit values, non-numeric engine tags, and distorted transaction outliers. After feature extraction and categorical encoding, linear benchmarks are evaluated against ensemble tree algorithms to determine the strongest predictive model for automated automotive appraisal.

## 🧹 Key Data Sanitization & Feature Engineering
- **String & Unit Parsing:** Stripped `' km'` suffixes from mileage, converted textual `'-'` levy entries into valid numeric tax records, and extracted binary `Turbo` indicators from composite engine capacity strings.
- **Categorical Normalization:** Cleaned spreadsheet auto-formatting artifacts in door counts (`04-May` $\rightarrow$ `4-5`) and mapped interior specifications into binary flags.
- **Outlier Filtering:** Removed non-market noise (prices under $1,000 and extreme multi-million dollar artifacts) alongside extreme mileage entries (>500,000 km) to preserve regression stability.
- **Temporal Transformation:** Derived `Vehicle_Age` from production year to capture true asset depreciation over calendar time.
- **Dimensionality Control:** Grouped low-frequency manufacturers into an `Other` tier prior to One-Hot Encoding to prevent matrix sparsity.

## ⚙️ Model Performance & Benchmarking

Models were evaluated on an 80/20 train-test split using $R^2$, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE):

| Model | Evaluation Role | $R^2$ Score | MAE ($) | RMSE ($) |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression (OLS)** | Baseline linear interpretability | ~0.35 - 0.40 | High | High |
| **Ridge Regression** | Collinearity regularization ($\alpha=1.0$) | ~0.35 - 0.40 | High | High |
| **Random Forest Regressor** | Non-linear interaction & tree ensemble | **~0.75 - 0.82** | **Lowest** | **Lowest** |

*Random Forest substantially outperformed linear models, successfully capturing complex non-linear depreciation curves and manufacturer-specific valuation premiums.*


## 📈 Top Valuation Drivers (Feature Importance)
1. **Engine Volume & Turbo:** Strongest direct indicator of vehicle trim, horsepower class, and base valuation.
2. **Vehicle Age (Depreciation):** Exponential depreciation driver; price decays most sharply across the first 3–5 years.
3. **Mileage (Asset Wear):** Inverse correlation with residual price, acting as the primary gauge of mechanical wear.
4. **Airbags & Safety Equipment:** Serves as a strong proxy for vehicle trim tier and build era.

### Requirements
- Python 3.9+
- Jupyter Notebook / VS Code / Google Colab

### Installation
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
