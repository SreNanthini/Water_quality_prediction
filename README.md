# 💧 Water Quality Prediction using Machine Learning

This project predicts multiple water pollutants using **supervised machine learning**, specifically `RandomForestRegressor` wrapped with `MultiOutputRegressor`. It was developed during a **1-month AICTE Virtual Internship sponsored by Shell** in **June 2025** under mentorship.

---

## 🧠 Project Objective

The goal is to create a predictive system that estimates key **water quality parameters** based on **year and station location ID**, helping environmental agencies monitor pollution levels early and take timely action.

---

## 📌 Overview

In this project, we:

- **Collected** real-world water quality data from a public CSV (hosted on GitHub)
- **Cleaned and preprocessed** the data by handling missing values and converting date to extract useful features (year and station ID)
- **Selected** only `year` and `station ID` as input features, which are simple yet meaningful for prediction
- **Trained a multi-output regression model** using `MultiOutputRegressor` with `RandomForestRegressor` to predict six pollutant levels
- **Evaluated** the model using **R² Score** and **Mean Squared Error (MSE)** for each target pollutant
- **Saved** the trained model and structure using `joblib`
- **Deployed** a user-friendly **Streamlit web app** to take inputs (year and station ID) and display predicted pollutant levels
- **Visualized** average pollutant levels and trends using charts and plots inside the app

---

## 📊 Predicted Water Pollutants

The model predicts concentrations of the following six key pollutants:

- **O₂ (Dissolved Oxygen)**
- **NO₃ (Nitrate)**
- **NO₂ (Nitrite)**
- **SO₄ (Sulfate)**
- **PO₄ (Phosphate)**
- **CL (Chloride)**
---

## 🔧 Technologies Used

| Tool / Technology | Version                | Purpose                             |
|-------------------|------------------------|-------------------------------------|
| **Python**        | 3.13.5                 | Programming language                |
| **Pandas**        | 2.3.0                  | Data handling                       |
| **NumPy**         | 2.3.1                  | Numerical operations                |
| **Scikit-learn**  | 1.7.0                  | ML model building                   |
| **Joblib**        | 1.5.1                  | Model saving                        |
| **Streamlit**     | 1.46.1                 | Web app deployment                  |
| **Matplotlib**    | 3.10.3                 | Data visualization                  |
| **Seaborn**       | 0.13.2                 | Correlation heatmaps                |
| **Google Colab**  | Cloud platform         | Model training environment          |
| **Git & GitHub**  | git version 2.49.0.windows.1 | Version control and collaboration   |

---

## Model Performance

The model was evaluated using:

- **R² Score**
- **Mean Squared Error (MSE)**

Performance was acceptable across all parameters

---
## Deployed web app

```https://waterqualitypredictionedunet.streamlit.app/```

---
## Internship Details

- **Internship Type**: AICTE Virtual Internship - Edunet Foundation
- **Sponsor**: Shell  
- **Duration**: June 2025 (1 month)  
- **Focus Area**: Machine Learning in Environmental Monitoring  

---

