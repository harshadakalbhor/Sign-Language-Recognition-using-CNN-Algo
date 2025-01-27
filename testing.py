import cv2
import numpy as np
import tensorflow as tf
import time

# Load the trained model
model = tf.keras.models.load_model('sign_language_cnn_model.h5')

# Define the label mapping based on your training
label_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'L'}
# Add more labels as necessary

# Function to preprocess frames
def preprocess_frame(frame):
    # Check if the frame is in grayscale; if so, convert to RGB (the model expects RGB)
    if frame.shape[2] == 1:  # Grayscale images have one channel
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    # Resize the frame to match the input size of the model (64x64)
    resized_frame = cv2.resize(frame, (64, 64))
    # Normalize pixel values
    normalized_frame = resized_frame / 255.0
    # Expand dimensions to match the model's input shape
    return np.expand_dims(normalized_frame, axis=0)

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break

    # Define the hand frame rectangle (x, y, width, height)
    hand_rect = (100, 100, 200, 200)  # Adjust this to best fit the hand placement
    
    # Draw the rectangle on the frame
    cv2.rectangle(frame, (hand_rect[0], hand_rect[1]), (hand_rect[0] + hand_rect[2], hand_rect[1] + hand_rect[3]), (0, 255, 0), 2)
    
    # Extract the region of interest (ROI) where the hand is supposed to be
    roi = frame[hand_rect[1]:hand_rect[1]+hand_rect[3], hand_rect[0]:hand_rect[0]+hand_rect[2]]
    
    # Preprocess the captured frame
    preprocessed_frame = preprocess_frame(roi)
    
    # Predict the gesture
    predictions = model.predict(preprocessed_frame)
    predicted_class = np.argmax(predictions)
    predicted_label = label_mapping[predicted_class]
    confidence = np.max(predictions) * 100  # Confidence percentage of the prediction

    # Display the resulting frame with the predicted gesture and confidence
    cv2.putText(frame, '{}: {:.2f}%'.format(predicted_label, confidence), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imshow('Sign Language Recognition', frame)
    
    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.01)  # Adding a slight delay to make it more real-time

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
