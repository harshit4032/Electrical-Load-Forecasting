#  Electrical Load Forecasting using ML, DL, and Hybrid Models

This repository presents a comprehensive approach to **short-term electricity load forecasting** using traditional Machine Learning (ML), Deep Learning (DL), and Hybrid models. The project was developed as part of the final-year research work at **NIT Kurukshetra**, aiming to optimize forecasting accuracy for national grid-scale power demand.

---

##  Objective

To design, evaluate, and compare multiple models for **hourly electricity demand prediction**, using time series data enriched with weather and calendar features. The goal is to identify models that minimize forecasting error and improve grid reliability.

---

##  Repository Structure

```
Electrical-Load-Forecasting/
├── data/                       # Raw and cleaned datasets
├── model_testing/                 
│   ├── ANN.ipynb        
│   ├── Arima.ipynb            
│   ├── CNN_LSTM.ipynb         
│   ├── DT.ipynb    
│   ├── GB.ipynb       
│   ├── LSTM.ipynb
│   ├── RF.ipynb 
│   ├── RF_XG_AR.ipynb 
│   ├── SVR.ipynb 
│   ├── TFT.ipynb 
│   ├── XGB.ipynb
├── EDA.ipynb              # Exploratory data analysis
├── SVR.ipynb              # Support Vector Regression
├── LSTM.ipynb             # LSTM model training & testing
├── GRU_CNN_model.ipynb    # GRU + CNN hybrid model
├── Evaluation.ipynb       # Model evaluation and comparison
├── results_processing.ipynb # Result aggregation and visualization
├── results/                   # Forecast vs actual plots and metrics
├── data
├── EDA
├── README.md                  # Project documentation
```

---

## Data Overview

* **Source**: Kaggle (Panama national grid dataset)
* **Granularity**: Hourly
* **Features**:

  * Historical Load
  * Temperature
  * Calendar-based indicators (weekday, weekend, holidays)
  * Lagged and rolling load statistics

---

## Models Implemented

### Traditional ML Models:

* Random Forest Regressor
* XGBoost
* Gradient Boosting
* SVR
* Decision Tree
* ARIMA

### Deep Learning Models:

* LSTM
* GRU
* CNN-LSTM

### Hybrid Models:

* ARIMA + MLP
* DWT + LSTM
* RF + XGBoost + ARIMA

---

##  Methodology

* **EDA**: Time series visualization, correlation, trend/seasonality decomposition
* **Feature Engineering**:

  * Lag features (24hr, 48hr)
  * Rolling mean and std
  * Hour of day, day of week, holiday flags
* **Model Training**:

  * Used `GridSearchCV` to tune ML models
  * Trained DL models using Keras with multiple input/output time windows
* **Evaluation**:

  * Evaluated over **14 distinct weekly test sets**
  * Metrics: MAE, RMSE, MAPE

---

## 📊 Results Summary

| Model             | MAE    | RMSE   | MAPE   |
| ----------------- | ------ | ------ | ------ |
| **Random Forest** | 10.61  | 13.93  | 0.89%  |
| XGBoost           | 15.78  | 20.29  | 1.31%  |
| SVR               | 33.24  | 41.94  | 2.75%  |
| GRU               | 38.96  | 47.50  | 3.21%  |
| LSTM              | 42.38  | 49.94  | 3.50%  |
| ARIMA             | 210.63 | 260.47 | 15.56% |

✅ Tree-based models outperformed both DL and hybrid setups.
📉 Hybrid ensemble (RF + XGB + ARIMA) underperformed due to poor compatibility.

---

## 📈 Sample Visualization

![Forecast vs Actual Plot](https://github.com/harshit4032/Electrical-Load-Forecasting/blob/main/results/ANN_actual_vs_predicted_logscale_Week%2001%2C%20Jan%202020.png)
![](https://github.com/harshit4032/Electrical-Load-Forecasting/blob/main/results/actual_vs_predicted_Week%2001%2C%20Jan%202020.png)
![Error](https://github.com/harshit4032/Electrical-Load-Forecasting/blob/main/results/error_metrics_Week%2001%2C%20Jan%202020.png)
*Comparison of predicted vs actual load for a selected week.*

---

## 🧾 Report

For detailed methodology, modeling rationale, and visualizations, refer to the full report:
📄 [`Report`](https://drive.google.com/file/d/1RBVjZEayu0sCG14feOGdGxZMdSH0nuQy/view?usp=sharing)

---

##  Tech Stack

* Python (pandas, numpy, scikit-learn, statsmodels)
* Keras (TensorFlow backend)
* Matplotlib / Seaborn
* Jupyter Notebook

---

##  Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/harshit4032/Electrical-Load-Forecasting.git
   cd Electrical-Load-Forecasting
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run any notebook in the `notebooks/` directory for model training, evaluation, or visualization.

---

## Author

**Harshit**
Electrical Engineering, NIT Kurukshetra
[GitHub](https://github.com/harshit4032)

---

