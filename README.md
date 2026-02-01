# PJME_MW-Prevision-

📊 Time Series Analysis & Forecasting System
Master ROMARIN | Techniques de prévision University of Science and Technology Houari Boumediene (USTHB)

📝 Project Overview
Developed under the supervision of Pr. Djamal Chaabane, this application provides a robust framework for analyzing and forecasting time series data. In today’s volatile economic environment, the tool automates the transformation of raw temporal data into actionable strategic insights.

🚀 Core Features
-Exploratory Data Analysis (EDA): Automatic detection of seasonality, trend analysis, and descriptive statistics.
-Stationarity Testing: Integrated ADF and KPSS tests to ensure data stability
-Forecasting Models: *Simple Exponential Smoothing.
                     *Holt’s Linear Trend Model.
                     *Holt-Winters Seasonal Model (Additive and Multiplicatives).

-Parameter Optimization: Parameters ($\alpha, \beta, \gamma$) are optimized by minimizing the Mean Squared Error (MSE) using Nelder-Mead or L-BFGS-B algorithms.
-Model Selection: Automatic selection based on AIC, BIC, and AICc criteria.

📋 Professional Output Log
The application generates a comprehensive session log for auditing purposes, including:

-Model Performance: Comparative tables featuring MSE, MAE, and MAPE.
-Residual Analysis: Normality (Shapiro-Wilk) and Autocorrelation (Ljung-Box) tests.
-Export Options: Results available in JSON, structured text, and synthetic PDF reports.


🛠️ Setup & Execution
1-Install dependencies:  pip install -r requirements.txt
2-Launch the application:  streamlit run app.py
