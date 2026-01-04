import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load the dataset
file_path = 'accident_dataset.csv'  # Update this path if required
data = pd.read_csv(file_path)

# Define features (X) and labels (y)
#data['longitude'] = data['longitude'].astype(float)
X = data[['Longitude', 'Latitude', 'Speed', 'Distance', 'Acc_X', 'Acc_Y', 'Acc_Z','Heading', 'gyro_x', 'gyro_y', 'gyro_z', 'label']].values
y = data['label'].values

# Clean and replace non-numeric data
def clean_and_convert_to_float(array):
    cleaned_array = []
    for row in array:
        cleaned_row = []
        for value in row:
            try:
                cleaned_row.append(float(value))  # Attempt to convert to float
            except ValueError:
                cleaned_row.append(np.nan)  # Replace non-numeric with NaN
        cleaned_array.append(cleaned_row)
    return np.array(cleaned_array)
# Clean data
X_cleaned = clean_and_convert_to_float(X)

# Remove rows with NaN
X_cleaned = X_cleaned[~np.isnan(X_cleaned).any(axis=1)]

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X_cleaned)
# Scale features
#scaler = StandardScaler()
#X = scaler.fit_transform(X)

# Reshape data for LSTM input: (samples, time_steps, features)
time_steps = 3  # Use 3 readings as one sequence
X_sequences = []
y_sequences = []
for i in range(len(X) - time_steps):
    X_sequences.append(X[i:i + time_steps])
    y_sequences.append(y[i + time_steps - 1])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

# Split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)

#create model
svm_model = SVC(kernel='rbf', probability=True, random_state=2)
#train model
svm_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
# Train the mode
# Evaluate
svm_predictions = svm_model.predict(X_test.reshape(X_test.shape[0], -1))
accuracy = accuracy_score(y_test, svm_predictions)
print(f"Random Forest Accuracy: {accuracy}")

# Compile the model
#rf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','precision', 'recall'])


# Save the model as HDF5
hdf5_model_path = "accident_detection_model.h5"
print(f"Model saved as {hdf5_model_path}")

