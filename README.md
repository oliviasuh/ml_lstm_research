# LSTM-based Time-Series Accident Prediction

## 1. Project Overview
This repository archives an independent machine learning research project
focused on accident prediction using time-series vehicle sensor data.

The study investigates whether sequence-based deep learning models
can outperform classical machine learning approaches
in safety-critical accident detection tasks.

## 2. Data Description
The dataset consists of multivariate time-series data collected from vehicle sensors.

Features include:
- GPS coordinates (Longitude, Latitude)
- Timestamp
- Acceleration (AccX, AccY, AccZ)
- Gyroscope data (GyroX, GyroY, GyroZ)
- Speed, Distance, Heading
- Binary label indicating accident occurrence (0 / 1)

Each sample represents a temporal sequence rather than an independent data point.

## 3. Data Processing
Preprocessing steps were designed to preserve temporal dependencies:
- Handling missing values
- Normalization of numerical features
- Binary encoding of accident labels
- Reshaping data into (samples, time steps, features) format for LSTM input

Flattening time-series features was intentionally avoided
to maintain sequential information critical for prediction.

## 4. Baseline Models
To establish a performance baseline, several classical machine learning models were implemented:
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

These models rely on handcrafted feature representations
and treat each data instance independently,
which limits their ability to model temporal patterns.

## 5. Proposed Model: LSTM
An LSTM-based neural network was implemented to capture long-term temporal dependencies
in sequential sensor data leading up to accident events.

The model architecture includes:
- Multiple stacked LSTM layers
- Dropout layers to mitigate overfitting
- A fully connected output layer for binary classification

LSTM was selected due to its effectiveness in modeling sequential patterns
and temporal correlations in time-series data.

## 6. Model Comparison & Evaluation
Models were evaluated using accuracy, precision, recall, and F1-score.
Given the safety-critical context, recall was emphasized
to minimize false negatives (missed accident cases).

Results indicate that:
- The LSTM model achieved the highest overall performance
  (Accuracy: 95.2%, Recall: 96.5%, F1-score: 94.1%)
- Classical models showed lower recall and weaker performance balance
- XGBoost performed best among classical approaches but still underperformed compared to LSTM

## 7. Discussion
Classical machine learning models demonstrate limitations
in handling sequential sensor data:
- SVM struggles with time-series representation
- Tree-based models perform reasonably but lack temporal learning capability
- Classical models train faster but fail to capture long-term dependencies

The LSTM model consistently outperformed classical approaches
by effectively learning temporal patterns preceding accident events.

## 8. Limitations & Future Work
- Dataset size limits model generalization
- Data was collected under constrained conditions
- Future work includes deployment with real-world user data
  and expansion to larger-scale datasets
