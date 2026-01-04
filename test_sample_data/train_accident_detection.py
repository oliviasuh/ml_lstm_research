import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load the dataset
file_path = 'accident_dataset.csv'  # Update this path if required
data = pd.read_csv(file_path)

# Define features (X) and labels (y)
X = data[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'speed']].values
y = data['label'].values

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

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

# Create an LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(time_steps, X.shape[1])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
    #Dense(64, activation='relu', input_shape=(8,)),  # Adjusted for 8 features
    #Dense(32, activation='relu'),
    #Dense(1, activation='sigmoid')  # Adjust for your output
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the model as HDF5
hdf5_model_path = "accident_detection_model.h5"
model.save(hdf5_model_path)
print(f"Model saved as {hdf5_model_path}")

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Disable lowering of tensor list ops
converter._experimental_lower_tensor_list_ops = False

# Use SelectTFOps if necessary
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

# Optionally, apply optimizations (e.g., for size reduction)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save the TensorFlow Lite model
tflite_model_path = "accident_detection_model.tflite"
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
print(f"Model converted and saved as {tflite_model_path}")
