# ml_lstm_research
This repo is for archieving ML LSTM algorithm research for indenepent research

1. Data source: Longitude, Latitude, Time, AccXYZ, Lable(0|1), Speed, Distance, Heading, GyroXYZ(time series date from vehicle sensor)
1. Data Processing
- Handle missing values.
- Normalize or standardize numerical features.
- Encode categorical labels (e.g., accident/no accident as binary values).
- Reshape data for LSTM (e.g., (samples, time steps, features) format
  
3. Baseline Models
To establish a performance baseline, classical machine learning models were implemented( SVM, Random Forest, XGBoost)

4. Proposed Model (LSTM-Long Short-term Memory)
An LSTM-based neural network was implemented to capture long-term temporal dependencies
in sensor data.

The model architecture included:
- Multiple LSTM layers
- Dropout for overfitting prevention
- Fully connected output layer for binary classification

LSTM was selected due to its ability to model sequential patterns
leading up to accident events.

5. Model Comparison & Evaluation
- LSTM Outperforms Classical ML Models: LSTM achieves the highest accuracy (95.2%), meaning it classifies accidents more correctly than classical models.
It also has the best recall (96.5%), indicating it detects more actual accident cases compared to other models.
- Precision vs Recall : LSTM has slightly lower precision (93.8%) than recall (96.5%)
A high recall means LSTM is better at detecting real accident cases, even if it occasionally raises false alarms.
- F1-Score Comparison: LSTM has the highest F1-score (94.1%), meaning it balances precision and recall the best.
Classical models struggle to achieve this balance, with Logistic Regression performing the worst.

6. Classical ML Models shows limitations.
- Random Forest & SVM rely on handcrafted feature extraction, making them less effective at capturing the sequential nature of accident-related data.
- Tree-based models (XGBoost, Random Forest) still perform decently, but lack temporal learning capabilities.
- SVM is not ideal for time-series sensor data due to its reliance on static features.
- Classical ML models are faster to train, but they struggle with sequential relationships, leading to lower accuracy and AUC scores.
- XGBoost is the best classical ML alternative but still does not outperform LSTM
  
7. Conclusion
- LSTM consistently outperforms classical machine learning models for time-series-based car accident detection due to its superior ability to learn temporal dependencies: Higher Accuracy(95.2%), Superior Recall (96.5%), Balanced F1-Score (94.1%): 


