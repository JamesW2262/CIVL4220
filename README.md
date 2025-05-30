# CIVL4220
CIVL4220FinalProject
# LSTM and SARIMA Hybrid Model for Hong Kong Electricity Consumption Forecasting

This repository contains the supplementary code for the report titled **"Exploring a Hybrid Model Combining LSTM and SARIMA for Forecasting Electricity Consumption in Hong Kong"**, submitted as part of the CIVL 4220 course at The Hong Kong Polytechnic University. The code implements a hybrid forecasting model that combines Long Short-Term Memory (LSTM) and Seasonal AutoRegressive Integrated Moving Average (SARIMA) techniques to predict electricity consumption across various categories in Hong Kong using monthly data from 1979.

## Overview

The Jupyter notebook (`LSTM_+_SARIMA_(sec_)ipynb.ipynb`) includes:
- **Data Preprocessing**: Loading and cleaning monthly electricity consumption data, handling anomalies with z-score-based outlier detection, and interpolating missing values.
- **Feature Engineering**: Adding lagged values (1-month and 12-month), rolling means (12-month window), and seasonal dummy variables (Spring, Summer, Winter).
- **Model Development**:
  - SARIMA model with parameter tuning using Akaike Information Criterion (AIC).
  - LSTM model with a Sequential architecture (three LSTM layers with 150 units, ReLU activation, and dropout).
  - Hybrid model combining SARIMA and LSTM forecasts with dynamic weighting based on cross-validated Mean Absolute Percentage Error (MAPE) scores.
- **Cross-Validation**: Time-series cross-validation (5 folds) to evaluate model performance (MAE, RMSE, MAPE).
- **Forecasting**: 10-year (120-month) forecasts for categories including Domestic, Commercial, Industrial, Street Lighting, and All Groups.
- **Visualization**: Plots comparing historical data with SARIMA, LSTM, and hybrid forecasts, plus bar charts of cross-validation metrics.

This code supports the methodology and results presented in the accompanying report, providing a reproducible implementation of the hybrid forecasting approach.

## Prerequisites

To run the code, ensure you have the following dependencies installed:
- **Python 3.x**
- **Libraries**:
  - `pandas` (for data manipulation)
  - `numpy` (for numerical operations)
  - `matplotlib` (for plotting)
  - `statsmodels` (for SARIMA modeling)
  - `scikit-learn` (for preprocessing and metrics)
  - `tensorflow` or `keras` (for LSTM modeling)

Install the required packages using pip:
```bash
pip install pandas numpy matplotlib statsmodels scikit-learn tensorflow
