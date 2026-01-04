# ml_lstm_research
This repo is for archieving ML LSTM algorithm research for indenepent research

1. Data source: Longitude, Latitude, Time, AccXYZ, Lable(0|1), Speed, Distance, Heading, GyroXYZ(time series date from vehicle sensor)
1. Data Processing
- Handle missing values.
- Normalize or standardize numerical features.
- Encode categorical labels (e.g., accident/no accident as binary values).
- Reshape data for LSTM (e.g., (samples, time steps, features) format
3. Baseline Models: SVM, Random Forest, XGBoost (classic model)
4. Proposed Model (LSTM-Long Short-term Memory)
5. Model Comparison & Evaluation
- LSTM Outperforms Classical ML Models: LSTM achieves the highest accuracy (95.2%), meaning it classifies accidents more correctly than classical models.
It also has the best recall (96.5%), indicating it detects more actual accident cases compared to other models.
- Precision vs Recall : LSTM has slightly lower precision (93.8%) than recall (96.5%)
A high recall means LSTM is better at detecting real accident cases, even if it occasionally raises false alarms.
- F1-Score Comparison: LSTM has the highest F1-score (94.1%), meaning it balances precision and recall the best.
Classical models struggle to achieve this balance, with Logistic Regression performing the worst.
