
# The extract_frames function is responsible to do the Pyramid and Sliding process accorss each video frame
# As a result, a

from resize import resize
import numpy as np
import threading
import cv2
import pandas as pd


class ExctractFrames(threading.Thread):
    def __init__(self, current_frame_queue, pyramid_queue,window_width, window_height, model_width, model_height , buffer_queue):
        super().__init__()
        self.current_frame_queue = current_frame_queue
        self.window_height = window_height
        self.window_width = window_width
        self.pyramid_queue = pyramid_queue
        self.buffer_queue = buffer_queue
        self.model_width = model_width
        self.model_height = model_height
        self.running = True

    def run(self):
        while self.running:
            # Creating a pyramid with decreasing window
            if not self.current_frame_queue.empty():
                #print('Pyramid running')
                self.current_frame = self.current_frame_queue.get()

                # The pyramid is to 1/2 of the window resolution (seted on top)
                while (self.current_frame.shape[0] >= self.window_height // 2
                       and
                       self.current_frame.shape[1] >= self.window_width // 2):
                    self.current_frame = cv2.pyrDown(self.current_frame)  # OpenCV pyramid
                    self.pyramid_queue.put(self.current_frame)  # Apending the extracted frame

                # Buffering (list) to accumulate the resulting ROI
                buffer = []

                # Loop on pyramid frames to slide and extract each ROI for future analysis
                self.pyramid = self.pyramid_queue.get()
                for level, pyramid_frame in enumerate(self.pyramid):
                    # Vertical slide
                    for y in range(0, pyramid_frame.shape[0] - self.window_height, self.window_height // 2):
                        # horizontal slide
                        for x in range(0, pyramid_frame.shape[1] - self.window_width, self.window_width // 2):
                            # The sliding window is limited by window_width/2
                            # This process will  be responsible to extract ROIs that will be used to find for objects

                            # Extract ROI
                            roi = pyramid_frame[y:y + self.window_height, x:x + self.window_width]

                            # Resize ROI using the resize function that keeps the correct proportion avoid shear
                            # Sending the ROI (frame) to resize function
                            roi_resized = resize(roi, self.model_width, self.model_height)
                            # Adding more one dimension to compatibilize with the model that is expacting an Array
                            # with multiple images, adding more one dimention will simplify it
                            roi_resized = np.expand_dims(roi_resized, axis=0)

                            # Store the ROI on the buffer keeping the original position of the roy for future extraction
                            # So, this position will be stored into a tupple: (x,y)
                            # into the "buffer list" like this for each line: [ (x,y), ROI_Frame ]
                            buffer.append(((x * (2 ** level), y * (2 ** level)), roi_resized))
                self.buffer_queue.put(buffer)

    def stop(self):
        self.running = False
        print("Stopping Extraction Frames Thread")


# This analyse_buffer function is responsible to receive the frame_buffer with all ROIs extracted in
# the function extract_frames

class AnalyseBuffer(threading.Thread):
    def __init__(self, buffer_queue, detections_queue, model, classes, detections_history, window_width, window_height,
                 model_width, model_height, frame_event):
        super().__init__()
        self.buffer_queue = buffer_queue
        self.model = model
        self.classes = classes
        self.detections_history = detections_history
        self.window_width = window_width
        self.window_height = window_height
        self.detections_queue = detections_queue
        self.model_width = model_width
        self.model_height = model_height
        self.frame_event = frame_event
        self.running = True

    def run(self):
        detections = []  # List of detections

        if self.running:
            while self.running:
                if not self.buffer_queue.empty():
                    print('Analisando')
                    buffer = self.buffer_queue.get()

                    # data concatenated data buffer, that was iterated on "buffer"
                    buffer_data = np.concatenate([item[1] for item in buffer], axis=0)
                    # Sending the concatenated buffer to the model and returning its predictions for each ROI
                    predictions = self.model.predict(buffer_data)

                    # Here, we are iterating on each "buffer" element
                    # and extrating the tupple "(x, y)" on the x_original and y_original
                    for i, ((x_original, y_original), _) in enumerate(buffer):
                        class_index = np.argmax(predictions[i])  # Getting the the prediction
                        class_id = self.classes[class_index][0]  # Getting the predicted class name

                        confidence = predictions[i, class_index]  # Geting the confidence

                        # If the confidence is above 0.35 we will verify if it persist in the next video frames and than,
                        # log it into detection buffer
                        if confidence > 0.35:

                            # Initialize or increment the counter for the class
                            self.detections_history[class_id] = self.detections_history.get(class_id, 0) + 1

                            # If the class reaches the frame detection threshold, add to the frame "detection" list
                            # At least 3 consecutive frames with the same detection
                            if self.detections_history[class_id] >= 2:

                                # Appending the ROI location, confidence and Class ID to the list
                                detections.append({
                                    'x_init': x_original,
                                    'y_init': y_original,
                                    'x_end': x_original + self.window_width,
                                    'y_end': y_original + self.window_height,
                                    'box_id': class_id,
                                    'confidence': confidence
                                })
                        else:
                            # If the confidence is less than 0.35 (defined treshold), reset the counter
                            self.detections_history[class_id] = 0

                    # return pd.DataFrame(detections)
                    # return the detections list
                    self.detections_queue.put(pd.DataFrame(detections))
                # Set a frame_event, indicating that the frame detection processing is done
                # even if no detection occurs
                self.frame_event.set()

    def stop(self):
        self.running = False
        print("Stopping Buffer Analyzer Thread")
