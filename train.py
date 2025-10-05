import pandas as pd
import sys
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger
from tqdm.keras import TqdmCallback

# --- 1. Define Parameters Directly ---
test_size = 0.2
random_state = 42
epochs = 50
batch_size = 16
dense_1_nodes = 16
dropout_rate = 0.2
dense_2_nodes = 8
optimizer = 'adam'

# --- 2. Load and Prepare the Dataset ---
data_path = os.path.join('Data', 'iris.csv') # Corrected path to 'Data/iris.csv'
model_path = 'iris_model.h5'
metrics_path = "metrics.csv"

# Column names for the Iris dataset
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_df = pd.read_csv(data_path)

print("Dataset loaded successfully.")

# Separate features (X) and labels (y)
X = iris_df.drop('species', axis=1)
y = iris_df['species']

# Convert categorical labels to numerical and then to one-hot encoding
y_encoded = pd.Categorical(y).codes
y_one_hot = to_categorical(y_encoded, num_classes=3)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y_one_hot, test_size=test_size, random_state=random_state
)

# Scale the features for better performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# --- 3. Define and Compile the Model ---
model = Sequential()
model.add(Dense(dense_1_nodes, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(dropout_rate))
model.add(Dense(dense_2_nodes, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- 4. Train the Model ---
print("\nStarting model training...")
model.fit(X_train, y_train,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(X_val, y_val),
          verbose=0,  # TQDM will handle the progress bar
          callbacks=[TqdmCallback(verbose=1), CSVLogger(metrics_path)])

print("Training finished.")

# --- 5. Save the Model ---
model.save(model_path)
print(f"Model saved to {model_path}")
