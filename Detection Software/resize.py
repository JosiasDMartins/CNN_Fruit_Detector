# Function to resize all images during the load process
def resize(image, width, height):
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

    # Returning the result
    return background_noise
