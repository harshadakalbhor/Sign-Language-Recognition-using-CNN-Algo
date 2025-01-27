import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load and preprocess the dataset
data_directory = 'data/'
images = []
labels = []

# Map labels to integers
label_mapping = {'A': 0, 'B': 1, 'C': 2, 'L': 3}  # Add more labels as necessary

# Iterate over subdirectories in the data directory
for label_dir in os.listdir(data_directory):
    label_dir_path = os.path.join(data_directory, label_dir)
    if os.path.isdir(label_dir_path):
        for filename in os.listdir(label_dir_path):
            # Load image
            image_path = os.path.join(label_dir_path, filename)
            image = Image.open(image_path)
            # Check if image is not in the expected mode (e.g., not RGB)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Resize image to a fixed size (e.g., 64x64)
            image = image.resize((64, 64))
            # Convert image to numpy array and normalize pixel values
            image = np.array(image) / 255.0
            # Append image to list
            images.append(image)
            # Append label to labels list
            labels.append(label_mapping[label_dir])  # Using directory name as label

# Convert lists to numpy arrays
X = np.array(images)
y = np.array(labels)

# Convert labels to one-hot encoding
y_one_hot = tf.keras.utils.to_categorical(y, num_classes=len(label_mapping))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Define the CNN model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(label_mapping), activation='softmax')  # Match the number of classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the trained model
model.save('sign_language_cnn_model.h5')
