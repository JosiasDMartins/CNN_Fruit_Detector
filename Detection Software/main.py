# This project is an experimental Object Detector for study purpose
# The idea is to use a CNN-trained model to identify a few fruit types
# into a video, marking the fruit with a bounding box using OpenCV
# The model was also trained by myself achieving a 92.85% accuracy
# This is not the best way to create an Object Detector, but is a good
# exercise to improve Python Skills and understand the basics of Object Detection
#
# NOTICE: This project was conceived to run with TensorFlow CUDA over GPU
#         It may run slowly on the CPU

# Dependencies:
# OpenCV
# Pandas
# Numpy
# TensorFlow 2.10 or newer

# Files:
# class_names csv file
# fruits.mp4 video
# fruits_model.h5 file

# video and h5 file can be downloaded from my GitHub
# Link: 
# Password: 

# Importing libraries
import cv2
import tensorflow as tf
import pandas as pd
from workers import ExctractFrames, AnalyseBuffer
from queue import Queue
from group_boxes import group_boxes

# Model definitions
width = 240
height = 240

# Video resolution definitions
video_width = 1280
video_height = 720

# Initial resolution of the window (ROI)
window_width = video_width//2  
window_height = video_height//2

# Video path
video_path = '../aditional_files/fruits.mp4'

# Model and classes
model = tf.keras.models.load_model('../aditional_files/fruits_model.h5')  # Loading the model
classes = pd.read_csv('../aditional_files/class_names.csv')  # Loading the classes names

# Printing classes DF
print(classes)

# Removing a duplicated column
classes = classes.drop(columns=['Unnamed: 0'])
print(classes)

# Converting to numpy array
classes = classes.to_numpy()

if __name__ == '__main__':

    # Creating a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # Checks if the video opened correctly
    if not cap.isOpened():
        print("Error oppening the video")
        exit()

    # Queues
    buffer_queue = Queue()
    pyramid_queue = Queue()
    current_frame_queue = Queue()
    detections_queue = Queue()
    detections_history = {}  # Creating an empty dictionary

    # Threads
    # Extracting frames / ROI using extract_frames function
    extract_frames = ExctractFrames(
        current_frame_queue,
        pyramid_queue,
        window_height,
        window_width,
        buffer_queue
    )
    analyse_buffer = AnalyseBuffer(
        buffer_queue,
        detections_queue,
        model,
        classes,
        detections_history,
        window_width,
        window_height
    )

    extract_frames.start()
    analyse_buffer.start()

    while True:
        # Capturing frames from the video
        ret, frame = cap.read()

        # If not, the video reach the end
        if not ret:
            break  # Break the software

        # Starting the image pyramid
        pyramid_queue.put([frame])
        current_frame_queue.put(frame)   # Saving the first frame

        # Evaluate the frame
        if not detections_queue.empty():
            # Analysing the buffer with analyse_buffer function
            boxes_coords = detections_queue.get()

            # Grouping all overlaped boxes
            boxes_grouped = group_boxes(boxes_coords)

            # Iterating on detections to plot the boxes
            for i in range(len(boxes_grouped)):
                # Extracting informations like boxes coords, label, confidence and if is grouped or not
                x_init = int(boxes_grouped['x_init'][i])
                y_init = int(boxes_grouped['y_init'][i])
                x_end = int(boxes_grouped['x_end'][i])
                y_end = int(boxes_grouped['y_end'][i])
                grouped = int(boxes_grouped['grouped'][i])
                label = boxes_grouped['box_id'][i]
                box_confidence = boxes_grouped['confidence'][i]

                if grouped:
                    # if grouped, the rectangle will be writed as 255 green
                    cv2.rectangle(frame, (x_init, y_init), (x_end, y_end), (0, 255, 0), 3)
                    cv2.putText(
                        frame,
                        f'{label}, '
                        f'{box_confidence:.2f}',
                        (x_init, y_init + 14),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
                else:
                    # if not grouped, the rectangle will ne writed as a 255 Blue
                    cv2.rectangle(frame, (x_init, y_init), (x_end, y_end), (255, 0, 0), 3)
                    cv2.putText(
                        frame,
                        f'{label}, {box_confidence:.2f}',
                        (x_init, y_init + 14),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2
                    )

        # Display the frame with stable detection boxes in real time
        cv2.imshow('Real-time Fruit Detection', frame)

        # Check if the 'q' key was pressed to close the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # stopping threads
    analyse_buffer.stop()
    extract_frames.stop()

    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()
