import tensorflow as tf
import pandas as pd
import numpy as np


# Load the trained model
model_path = "accident_detection_model.h5"  # Replace with your actual .h5 model path
model = tf.keras.models.load_model(model_path)
print(f"Model loaded successfully from {model_path}")
model.summary()
print(model.input_shape)#(None, 3, 12)

# Load the dataset
csv_path = "accident_dataset.csv"  # Replace with your actual dataset path
try:
    dataset = pd.read_csv(csv_path)
    print(f"Dataset loaded successfully from {csv_path}")
except FileNotFoundError:
    print(f"Dataset file not found at {csv_path}")
    exit()

def duplicate_features(data, target_size=21):
    """
    Duplicates features until the number matches target_size.
    """
    while data.shape[1] < target_size:
        data = np.hstack([data, data[:, :target_size - data.shape[1]]])
    return data

# Preprocess the dataset
def preprocess_data(df):
    """
    Preprocess the input data to match the model's expected input shape.
    
    Parameters:
        df (pd.DataFrame): Input dataframe containing the features.
    
    Returns:
        np.ndarray: Preprocessed features with the shape (num_samples, 3, 7).
    """
    # Assuming the target column is named 'label', drop it if present
    if 'label' in df.columns:
        df = df.drop(columns=['label'])

    # Check the total number of features
    total_features = df.shape[1]
    print(f"total_feature={total_features}")

    # Ensure the total number of features matches the model's expected input size (3 * 7 = 21)
#    if total_features != 3 * 7:
#        raise ValueError(f"Dataset features ({total_features}) do not match the required shape (3 * 7).")

    df = duplicate_features(dataset.to_numpy())
    # Reshape data into shape (num_samples, 3, 7)
    reshaped_data = df.reshape(-1, 3, 7)

    return reshaped_data

# Preprocess the dataset for inference
try:
    input_data_temp = preprocess_data(dataset)
    print(f"Input data preprocessed successfully: {input_data_temp.shape}")
except ValueError as e:
    print(e)
    exit()
#################### TEST

def preprocess_input(input_data):

    # Add 5 zero-padded features
    current_features = 7
    required_features = 12
    additional_features = required_features - current_features
    # Pad input_data along the features dimension
    input_data_padded = np.pad(
        input_data,
        pad_width=((0, 0), (0, 0), (0, additional_features)),
        mode='constant',
        constant_values=0
    )
    # Clean non-numeric values
    input_data_cleaned = np.empty_like(input_data_padded, dtype=float)
    for i in range(input_data_padded.shape[0]):
        for j in range(input_data_padded.shape[1]):
            for k in range(input_data_padded.shape[2]):
                try:
                    input_data_cleaned[i, j, k] = float(input_data_padded[i, j, k])
                except ValueError:
                    input_data_cleaned[i, j, k] = np.nan  # Replace invalid entries with NaN

    # Remove rows with NaN
    valid_rows = ~np.isnan(input_data_cleaned).any(axis=(1, 2))
    input_data_cleaned = input_data_cleaned[valid_rows]
    
    # Convert to float32
    return input_data_cleaned.astype("float32")

# Preprocess your data
input_data_cleaned = preprocess_input(input_data_temp)

########################
# Perform inference
print("Performing inference on the dataset...")
predictions = model.predict(input_data_cleaned)
print(f"Inference completed. Predictions shape: {predictions.shape}")

# Postprocess the predictions (example assumes binary classification)
if predictions.shape[1] == 1:  # Binary classification
    predicted_classes = (predictions > 0.5).astype(int)
    print("Binary classification predictions:")
    print(predicted_classes.flatten())
else:  # Multiclass classification
    predicted_classes = np.argmax(predictions, axis=1)
    print("Multiclass classification predictions:")
    print(predicted_classes)

# Save the predictions to a new CSV file
output_csv_path = "accident_predictions.csv"
output_df = pd.DataFrame({
    "Prediction": predicted_classes.flatten()
})
output_df.to_csv(output_csv_path, index=False)
print(f"Predictions saved to {output_csv_path}")

