#This project is an experimental Object Detector for study purpose
#The idea is to use a CNN-trained model to identify a few fruit types
#into a video, marking the fruit with a bounding box using OpenCV
#The model was also trained by myself achieving a 92.85% accuracy
#This is not the best way to create an Object Detector, but is a good
#exercise to improve Python Skills and understand the basics of Object Detection
#
# NOTICE: This project was conceived to run with TensorFlow CUDA over GPU
#         It may run slowly on the CPU

#Dependencies:
# OpenCV
# Pandas
# Numpy
# TensorFlow 2.10 or newer

#Files:
# class_names csv file
# fruits.mp4 video
# fruits_model.h5 file

# video and h5 file can be downloaded from my GitHub
# Link: 
# Password: 

#Importing libraries
import cv2
import tensorflow as tf
import pandas as pd
import numpy as np

#Model definitions
width = 240
height = 240

#Video resolution definitions
video_width = 1280
video_height = 720

#Initial resolution of the window (ROI)
window_width = video_width//2  
window_height = video_height//2

#Video path
video_path = 'fruits.mp4'  

#Model and classes
model = tf.keras.models.load_model('fruits_model.h5') #Loading the model
classes = pd.read_csv('class_names.csv')  #Loading the classes names

#Printing classes DF
print(classes)

#Removing a duplicated column
classes = classes.drop(columns=['Unnamed: 0'])
print(classes)

#Converting to numpy array
classes = classes.to_numpy()

#Function to resize all images during the load process

def resize(image):
    # Define the desired width and height
    desired_width, desired_height = width, height

    # Get the dimensions of the original image
    local_height, local_width = image.shape[:2]

    # Calculate the resizing ratio
    width_ratio = desired_width / local_width
    height_ratio = desired_height / local_height

    # Choose the minimum ratio to ensure the image fits within the new size
    ratio = min(width_ratio, height_ratio)

    # Calculate the new dimensions of the image
    new_width = int(local_width * ratio)
    new_height = int(local_height * ratio)

    # Resize the image to the new dimensions
    resized_image = cv2.resize(image, (new_width, new_height))

    # Create a blank image with the desired size and random background colors (noise)
    background_noise = np.random.randint(0, 256, (desired_height, desired_width, 3), dtype=np.uint8)

    # Calculate the starting coordinates to paste the resized image in the center
    x_start = (desired_width - new_width) // 2
    y_start = (desired_height - new_height) // 2

    # Paste the resized image in the center of the noisy background
    background_noise[y_start:y_start + new_height, x_start:x_start + new_width] = resized_image

    #Returning the result
    return background_noise 
    
#The extract_frames function is responsible to do the Pyramid and Sliding process accorss each video frame
#As a result, a

def extract_frames(current_frame):
   # Creating a pyramid with decreasing window
    while current_frame.shape[0] >= window_height //2 and current_frame.shape[1] >= window_width // 2:  #The pyramid is to 1/2 of the window resolution (seted on top)
        current_frame = cv2.pyrDown(current_frame)  #OpenCV pyramid
        pyramid.append(current_frame)  #Apending the extracted frame

    # Buffering (list) to accumulate the resulting ROI
    buffer = []

    # Loop on pyramid frames to slide and extract each ROI for future analysis
    for level, pyramid_frame in enumerate(pyramid):
        for y in range(0, pyramid_frame.shape[0] - window_height, window_height // 2):  #Vertical slide
            for x in range(0, pyramid_frame.shape[1] - window_width, window_width // 2):  #horizontal slide
                #The sliding window is limited by window_width/2 
                #This process will  be responsible to extract ROIs that will be used to find for objects
                
                # Extract ROI
                roi = pyramid_frame[y:y + window_height, x:x + window_width]

                # Resize ROI using the resize function that keeps the correct proportion avoid shear
                roi_resized = resize(roi) #Sending the ROI (frame) to resize function
                #Adding more one dimension to compatibilize with the model that is expacting an Array
                #with multiple images, adding more one dimention will simplify it
                roi_resized = np.expand_dims(roi_resized, axis=0) 

                # Store the ROI on the buffer keeping the original position of the roy for future extraction
                # So, this position will be stored into a tupple: (x,y) 
                # into the "buffer list" like this for each line: [ (x,y), ROI_Frame ]
                buffer.append(((x * (2 ** level), y * (2 ** level)), roi_resized))  
    return buffer

#This analyse_buffer function is responsible to receive the frame_buffer with all ROIs extracted in
#the function extract_frames

def analyse_buffer(buffer):
    detections = []  #List of detections

    #data concatenated data buffer, that was iterated on "buffer"    
    buffer_data = np.concatenate([item[1] for item in buffer], axis=0)
    predictions = model.predict(buffer_data)  #Sending the concatenated buffer to the model and returning its predictions for each ROI

    #Here, we are iterating on each "buffer" element and extrating the tupple "(x, y)" on the x_original and y_original  
    for i, ((x_original, y_original), _) in enumerate(buffer):
        class_index = np.argmax(predictions[i])   #Getting the the prediction
        class_id = classes[class_index][0]  #Getting the predicted class name
        
        confidence = predictions[i, class_index] #Geting the confidence
        if confidence > 0.35:  #If the confidence is above 0.35 we will verify if it persist in the next video frames and than, log it into detection buffer
                                    
            # Initialize or increment the counter for the class
            detections_history[class_id] = detections_history.get(class_id, 0) + 1
            
            # If the class reaches the frame detection threshold, add to the frame "detection" list
            if detections_history[class_id] >= 2:  #At least 3 consecutive frames with the same detection

                #Appending the ROI location, confidence and Class ID to the list
                detections.append({  
                    'x_init': x_original,
                    'y_init': y_original,
                    'x_end': x_original + window_width,
                    'y_end': y_original + window_height,
                    'box_id': class_id,
                    'confidence': confidence
                })
        else:
            # If the confidence is less than 0.35 (defined treshold), reset the counter
            detections_history[class_id] = 0

    return pd.DataFrame(detections)   #return the detections list    
    
#This function is responsible to group overlaped boxes to simplify the visualization

def group_boxes(boxes):
    boxes_agrup = []  #boxes agrupation list
    if len(boxes) > 0:   #If has detections, start the group analysis process
        
        #Data dictionary
        data = {
            'x_init': [],
            'y_init': [],
            'x_end': [],
            'y_end': [],
            'box_id': [],
            'grouped': 0,
            'confidence': []  
        }
        #Converting the dictionary into a PD Dataframe
        boxes_agrup = pd.DataFrame(data)

        #Getting a list os unique labels (classes) present in "boxes"
        labels = boxes['box_id'].unique().tolist()

        #Iterating on labels
        for label in labels:
            #Extrating all "equal label" in each "labels" iterations
            #If we have a labels list with "orange, grape, orange", as a result we will have a list with "orange, orange" 
            #in the first iteration and "grape" in the second iteration
            filtered = boxes[boxes['box_id'] == label]   #Filtering "label" on "boxes"
            overlap_boxes = []  #overlap list
            confidence_values = []  #Confidence of each detection list

            #Iterating on the filtered labels
            for i in range(1, len(filtered)):  #Geting the length to iterate on each "filtered" element, sraering in 1                
                for j in range(i): #Iterating in range of i, starting in 0 to realize a combinational analysis
                    box_i = filtered.iloc[i] #Getting the "i", that is the START position of the ROI
                    box_j = filtered.iloc[j] #gerring the "j", that is the END position of the ROI

                    #Overlap analysis - Verifying if has any overlap between the start/end of each "filtered" box
                    overlap = (
                        (box_i['x_end'] > box_j['x_init']) and (box_i['x_init'] < box_j['x_end']) and
                        (box_i['y_end'] > box_j['y_init']) and (box_i['y_init'] < box_j['y_end'])
                    )

                    #If any overlap is detected
                    if overlap:
                        #Extending "overlap_boxes" list with the position of the overlaped ROI
                        overlap_boxes.extend([box_i, box_j])
                        #Storing the confidence for future calculation
                        confidence_values.extend([box_i['confidence'], box_j['confidence']])

            #If any overlaped boxes is detected
            if overlap_boxes:
                # Find the minimum and maximum coordinates to span all overlapping boxes
                #Here, we are using min/max to get the min/max values, returning the x_init/y_init through a lambda function called "x"
                #As a result, min/max will find dictnary with the min/max, for specified key (x_init, y_init etc)
                #and we need to get the specific value 
                x_init_min = min(overlap_boxes, key=lambda x: x['x_init'])['x_init'] #Getting the x_init min
                y_init_min = min(overlap_boxes, key=lambda x: x['y_init'])['y_init'] #Getting the y_init min
                x_end_max = max(overlap_boxes, key=lambda x: x['x_end'])['x_end'] #Getting the x_init max
                y_end_max = max(overlap_boxes, key=lambda x: x['y_end'])['y_end'] #Getting the y_init max
                
                # Calculate the average of confidences of the overlapped boxes
                mean_confidence = sum(confidence_values) / len(confidence_values)

                #Concatenating  the results into a PD Dataframe to have a list of grouped boxes
                boxes_agrup = pd.concat([boxes_agrup, pd.DataFrame([{'x_init': x_init_min, 'y_init': y_init_min, 'x_end': x_end_max, 'y_end': y_end_max, 'box_id': label, 'grouped': 1, 'confidence': mean_confidence}])], ignore_index=True)
            else:
                # Include boxes that did not overlap in the final DataFrame
                boxes_agrup = pd.concat([boxes_agrup, filtered.assign(grouped=0)], ignore_index=True)

    return boxes_agrup


# Creating a VideoCapture object
cap = cv2.VideoCapture(video_path)

# Checks if the video opened correctly
if not cap.isOpened():
    print("Error oppening the video")
    exit()


while True:
    # Capturing frames from the video
    ret, frame = cap.read()

    #If not, the video reach the end
    if not ret:
        break  #Break the software

    # Starting the image pyramid
    pyramid = [frame]  #Pyramid list started with the first frame
    current_frame = frame  #Saving the first frame

    #Extracting frames / ROI using extract_frames function
    buffer = extract_frames(current_frame)

    # Evaluate the frame
    if buffer:
        detections_history = {}  #Creating an empty dictionary

        #Analysing the buffer with analyse_buffer function
        boxes_coords = analyse_buffer(buffer)
        #Grouping all overlaped boxes
        boxes_grouped = group_boxes(boxes_coords)  
        
        #Iterating on detections to plot the boxes    
        for i in range(len(boxes_grouped)):
            #Extracting informations like boxes coords, label, confidence and if is grouped or not
            x_init = int( boxes_grouped['x_init'][i]) 
            y_init =int(boxes_grouped['y_init'][i])
            x_end = int(boxes_grouped['x_end'][i])
            y_end = int(boxes_grouped['y_end'][i])  
            grouped = int(boxes_grouped['grouped'][i])
            label = boxes_grouped['box_id'][i]
            box_confidence = boxes_grouped['confidence'][i]
                      
            if grouped:
                #if grouped, the rectangle will be writed as 255 green
                cv2.rectangle(frame, (x_init,y_init), (x_end,y_end), (0,255,0), 3)
                cv2.putText(frame, f'{label}, {box_confidence:.2f}', (x_init, y_init + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:          
                #if not grouped, the rectangle will ne writed as a 255 Blue
                cv2.rectangle(frame, (x_init,y_init), (x_end,y_end), (255,0,0), 3)
                cv2.putText(frame, f'{label}, {box_confidence:.2f}', (x_init, y_init + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Display the frame with stable detection boxes in real time
    cv2.imshow('Real-time Fruit Detection', frame)

    # Check if the 'q' key was pressed to close the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()