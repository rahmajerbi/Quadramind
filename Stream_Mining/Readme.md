## SBP_DBP_Forecasting.ipynb

This Jupyter Notebook (`Multi_Step_SBP_DBP_Forecasting.ipynb`) contains the code used for building the forecasting model to predict the next 30 minutes of Systolic Blood Pressure (SBP) and Diastolic Blood Pressure (DBP) based on historical data. Using a Long Short-Term Memory (LSTM) neural network, the model leverages the past 10 minutes of blood pressure measurements to make accurate and timely predictions.


## forecasting_consumer_producer.ipynb

The `forecasting_consumer_producer.ipynb` notebook covers the real-time aspect of the forecasting. It includes code for consuming input data from a Kafka topic, producing predictions, and pushing the results back to another Kafka topic. Each minute the forecasting is done for the next 10 minutes based on incoming real-time data.
