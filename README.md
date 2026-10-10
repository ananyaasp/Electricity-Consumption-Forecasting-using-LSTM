# Electricity Consumption Forecasting Using Weather and Historical Data

## Overview

This project uses machine learning and time-series analysis to forecast electricity consumption based on historical usage patterns and weather conditions. An LSTM (Long Short-Term Memory) neural network is trained to learn temporal dependencies and predict future electricity demand.

The project explores short-term forecasting and recursive multi-step forecasting to understand how electricity demand predictions can support energy planning and consumption monitoring.

## Objectives

- Forecast electricity consumption using historical data.
- Incorporate weather features such as temperature, humidity, and wind speed.
- Learn temporal patterns using an LSTM neural network.
- Evaluate prediction performance using MAE, RMSE, and MAPE.
- Generate future consumption forecasts for different time horizons.

## Dataset

The dataset, `smart_meter_data.csv`, contains smart-meter electricity consumption records and associated weather features.

| Column | Description |
|---|---|
| `Timestamp` | Date and time of the observation |
| `Electricity_Consumed` | Target variable to predict |
| `Temperature` | Temperature measurement |
| `Humidity` | Humidity measurement |
| `Wind_Speed` | Wind speed measurement |
| `Avg_Past_Consumption` | Historical consumption feature |
| `Anomaly_Label` | Label used to identify anomalous records |

**Preprocessing steps:**
- Convert timestamps to datetime format and sort records chronologically.
- Retain records labelled `Normal` for the forecasting experiment.
- Remove the anomaly-label column from the model input data.
- Split the data chronologically into 80% training data and 20% testing data.

The dataset is not included in this repository. Add the CSV file locally before running the notebook, subject to the dataset's licensing and redistribution terms.

## Technologies Used

- **Python** — Data processing and model development
- **Pandas & NumPy** — Data manipulation and sequence preparation
- **TensorFlow / Keras** — LSTM model implementation
- **Scikit-learn** — Evaluation metrics
- **Matplotlib** — Forecast visualization
- **Jupyter Notebook** — Experimentation and analysis

## Methodology

### 1. Data Preparation

The data is sorted chronologically to preserve temporal order. A sequential 80:20 train-test split is used instead of random splitting.

### 2. Feature Engineering

The model uses the following four input features:

- Temperature
- Humidity
- Wind speed
- Average past consumption

A sliding window of 24 time steps is created for each prediction. Assuming observations occur every 30 minutes, the model uses the previous 12 hours of features to predict the next consumption value.

### 3. LSTM Model Architecture

The model consists of:

- LSTM layer with 64 units and `tanh` activation
- Dropout layer with a rate of 0.2
- LSTM layer with 32 units
- Dropout layer with a rate of 0.2
- Dense layer with 16 units and ReLU activation
- Output layer with one neuron for consumption regression

**Training configuration:**

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss function | Mean Squared Error (MSE) |
| Evaluation metric during training | Mean Absolute Error (MAE) |
| Epochs | 25 |
| Batch size | 32 |
| Input sequence length | 24 time steps |
| Training split | 80% |
| Testing split | 20% |

### 4. Short-Term Forecasting

The trained model predicts electricity consumption for the next time step using a sequence of the previous 24 observations.

Predictions are compared with actual test-set values using evaluation metrics and an actual-versus-predicted visualization.

### 5. Multi-Step Forecasting

Recursive forecasting is used to generate predictions over longer horizons, including:

- 24 hours
- 7 days
- 60 days

For these experiments, each predicted value is fed into the next input sequence. The notebook also experiments with simulated variations in future weather features.

**Important:** The longer-horizon forecasts are exploratory. Simulated weather variations are not actual weather forecasts, and recursive prediction errors may accumulate. Reliable long-term forecasting would require better future weather inputs and validation over multiple forecast horizons.

## Model Evaluation

The notebook's recorded test-set results are:

| Metric | Result |
|---|---:|
| Root Mean Squared Error (RMSE) | 0.1491 |
| Mean Absolute Error (MAE) | 0.1215 |
| Mean Absolute Percentage Error (MAPE) | 65.00% |

These values are in the target variable's current scale.

- **MAE:** Measures the average absolute prediction error.
- **RMSE:** Penalizes larger prediction errors more heavily.
- **MAPE:** Measures percentage error relative to actual consumption and can be sensitive to zero or very small target values.

The reported MAPE excludes zero-valued actual observations. The results should be treated as an initial baseline rather than proof of high forecasting accuracy. Further validation and comparison against simpler forecasting models are needed.

## Visualizations

The notebook generates visualizations for:

- Actual versus predicted electricity consumption on the test set
- Forecasted consumption for future time horizons
- Training and validation loss, where plotted

These plots help inspect prediction quality, forecast stability, and potential limitations of recursive forecasting.

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/electricity-consumption-forecasting.git
cd electricity-consumption-forecasting
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow jupyter
```

### 3. Add the dataset

Place `smart_meter_data.csv` in the project directory, alongside the notebook.

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook and execute the cells in order, starting with data preparation and ending with the forecasting experiments.

## Future Improvements

- Compare LSTM performance with ARIMA/SARIMAX, Prophet, and tree-based regression models.
- Add time-based features such as hour of day, day of week, and seasonality.
- Use a dedicated validation set and preserve the test set for final evaluation.
- Improve multi-step forecasting with direct multi-output prediction.
- Incorporate real future weather forecasts rather than simulated weather variations.
- Compare against persistence and seasonal-naive baselines.
- Evaluate performance separately for 30-minute, 24-hour, 7-day, and longer forecast horizons.
- Investigate forecast uncertainty and prediction intervals.

## Conclusion

This project demonstrates an end-to-end LSTM forecasting workflow, from time-series preprocessing and sequence generation to model training, evaluation, and recursive future prediction. It provides a starting point for exploring the relationship between weather conditions, historical consumption, and electricity demand.

