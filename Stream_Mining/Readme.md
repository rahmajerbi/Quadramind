## Multi_Step_SBP_DBP_Forecasting.ipynb
This Jupyter Notebook (`Multi_Step_SBP_DBP_Forecasting.ipynb`) contains the code used for building the forecasting model to predict the next 30 minutes of Systolic Blood Pressure (SBP) and Diastolic Blood Pressure (DBP). Using a Long Short-Term Memory (LSTM) neural network, the model leverages the past 10 minutes of blood pressure measurements to make accurate and timely predictions.


## Forecasting_Consumer.ipynb
The `Forecasting_Consumer.ipynb` notebook covers the real-time aspect of the forecasting. It includes code for consuming input data from a Kafka topic, producing predictions. Each minute the forecasting is done for the next 30 minutes based on incoming real-time data.

## IsolationForest_AnomalyDetection.ipynb
The `IsolationForest_AnomalyDetection.ipynb` notebook showcases the use of the Isolation Forest algorithm for spotting anomalies in Systolic Blood Pressure (SBP) and Diastolic Blood Pressure (DBP) readings. It focuses on identifying irregularities or outliers within the blood pressure data, employing Isolation Forest as a specialized technique designed for this purpose.

## consumer_IF_AD.ipynb
The `consumer_IF_AD.ipynb` notebook details a real-time process for detecting anomalies. It involves consuming data from a Kafka topic, using the Isolation Forest algorithm to detect anomalies within physiological signals, and then forwarding the identified anomalies to another Kafka topic. This notebook sets up a continuous analysis of incoming real-time data, swiftly identifying anomalies in the physiological signals for immediate action.

## SPO2_Monitoring.ipynb
The `SPO2_Monitoring.ipynb` notebook includes code for continuously consuming SPO2 data from a Kafka topic, storing this data in InfluxDB, and monitoring and logging the lowest observed SPO2 level over time.


## Dash_Plotly_Forecasting.ipynb
This `Dash_Plotly_Forecasting.ipynb` notebook creates a Dash web application that visualizes predictions for Systolic Blood Pressure (SBP) and Diastolic Blood Pressure (DBP) received through a Kafka topic. Plotly's Dash is an open-source framework for building data visualization interfaces.





