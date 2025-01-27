import cv2
import os

# Define the root directory to store collected data
root_directory = 'data/'

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set the resolution of the captured video
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define the labels for different signs and their corresponding keys
labels_keys = {
    'A': ord('a'),
    'B': ord('b'),
    'C': ord('c'),
    'L': ord('l')
    # Add more labels and keys as needed
}

# Initialize variables
current_label = None
current_gesture_count = 0
max_gesture_count = 100  # Number of samples to collect for each label

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define the hand frame rectangle (x, y, width, height)
    hand_rect = (100, 100, 200, 200)  # Adjust as needed to fit the hand region

    # Draw the rectangle on the frame
    cv2.rectangle(frame, (hand_rect[0], hand_rect[1]), (hand_rect[0] + hand_rect[2], hand_rect[1] + hand_rect[3]), (0, 255, 0), 2)

    # Display the current frame
    cv2.imshow('Collecting Data', frame)

    # Check for key press
    key = cv2.waitKey(1)

    # If 'q' is pressed, quit the program
    if key == ord('q'):
        break

    # If a valid key is pressed, set the current label
    if key in labels_keys.values():
        current_label = [label for label, keycode in labels_keys.items() if keycode == key][0]
        # Ensure a directory exists for the current label
        label_directory = os.path.join(root_directory, current_label)
        if not os.path.exists(label_directory):
            os.makedirs(label_directory)
        current_gesture_count = 0  # Reset count whenever a new label is selected

    # If the current label is set and the maximum number of samples hasn't been reached
    if current_label is not None and current_gesture_count < max_gesture_count:
        # Save the current frame as an image
        image_filename = f'{current_label}_{current_gesture_count}.jpg'
        image_path = os.path.join(label_directory, image_filename)

        # Extract the region of interest (ROI) where the hand is supposed to be
        roi = frame[hand_rect[1]:hand_rect[1] + hand_rect[3], hand_rect[0]:hand_rect[0] + hand_rect[2]]

        # Save the ROI as an image
        cv2.imwrite(image_path, roi)
        print(f'Saved: {image_path}')

        # Increment the gesture count for the current label
        current_gesture_count += 1

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
