import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load predictions from the file
file_path = "accident_predictions.csv"
data = pd.read_csv(file_path)

# Extract predictions
y_pred = data['Prediction']  # Ensure this column exists in the file

# Simulate ground truth (for example: alternating 0 and 1)
# Replace this with actual ground truth if available
import numpy as np
y_true = np.tile([0, 1], len(y_pred) // 2 + 1)[:len(y_pred)]

# Compare predictions with simulated ground truth
print("=== Performance Metrics ===")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))

# Optional: Check prediction statistics
print("\nPrediction Summary:")
print(data['Prediction'].value_counts())

