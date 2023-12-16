## SBP_DBP_Forecasting.ipynb

This Jupyter Notebook (`SBP_DBP_Forecasting.ipynb`) contains the code used for building the LSTM model for time series forecasting. It covers data preprocessing, model training, and evaluation for forecasting SBP and DBP values.

## prediction_utils.py

The `prediction_utils.py` file contains a modular utility function `predict_future` that predicts future values for SBP and DBP based on a trained model and input data.

## forecasting_consumer_producer.ipynb

The `forecasting_consumer_producer.ipynb` notebook covers the real-time aspect of the forecasting. It includes code for consuming input data from a Kafka topic, producing predictions, and pushing the results back to another Kafka topic. Each minute the forecasting is done for the next 10 minutes based on incoming real-time data.
