import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score

# Assuming your predictions are stored in 'predictions' and the true labels in 'y_true'
# For classification models, 'predictions' and 'y_true' will be discrete class labels
# For regression models, 'predictions' and 'y_true' will be continuous values

# Example: Load your data or predictions
#predictions = np.array([0, 1, 0, 1, 0])  # Predicted labels (for classification)
# Load predictions from the file
file_path = "accident_predictions.csv"
data = pd.read_csv(file_path)

#Extract Predtions
predictions = data['Prediction']  # Ensure this column exists in the file

y_true = np.tile([0, 1], len(predictions) // 2 + 1)[:len(predictions)]

# -- For Classification Models --

# 1. Basic accuracy
accuracy = accuracy_score(y_true, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# 2. Confusion matrix
conf_matrix = confusion_matrix(y_true, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# 3. Classification report (Precision, Recall, F1-score)
class_report = classification_report(y_true, predictions)
print("Classification Report:")
print(class_report)

# -- For Regression Models --

# In case you have a regression model with continuous predictions and actual values:

# 1. Mean Squared Error (MSE)
mse = mean_squared_error(y_true, predictions)
print(f"Mean Squared Error (MSE): {mse:.2f}")

# 2. R-squared (R2)
r2 = r2_score(y_true, predictions)
print(f"R-squared (R2): {r2:.2f}")

# -- Visual Analysis (For Both Classification and Regression) --

# Example: Predicted vs Actual (for regression or classification visual analysis)
plt.figure(figsize=(10,6))

# If it's a classification problem:
plt.subplot(1, 2, 1)
plt.scatter(np.arange(len(y_true)), y_true, color='b', label='True Labels')
plt.scatter(np.arange(len(predictions)), predictions, color='r', label='Predicted Labels')
plt.title('Predictions vs True Labels (Classification)')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.legend()

# If it's a regression problem:
plt.subplot(1, 2, 2)
plt.scatter(y_true, predictions, color='g')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='blue', lw=2)
plt.title('Predictions vs True Values (Regression)')
plt.xlabel('True Values')
plt.ylabel('Predictions')

plt.tight_layout()
plt.savefig('prediction_analysis_plot.png')
#plt.switch_backend('TkAgg')
#plt.show()

# Optional: Save analysis to a report file
analysis_report = {
    "Accuracy": accuracy,
    "Confusion Matrix": conf_matrix,
    "Classification Report": class_report,
    "Mean Squared Error": mse,
    "R-squared": r2,
}

# Save to CSV
pd.DataFrame.from_dict(analysis_report, orient='index').to_csv('prediction_analysis.csv')

# For more detailed visualizations, you can use seaborn's heatmap or other advanced plots.

