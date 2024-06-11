import skvideo.io
import tensorflow as tf
import numpy as np

def crop_single(frames, coordinates, target_dimension):
    """
    Crop one sample's frames around the specified coordinates to the target dimensions.

    Args:
    frames (tf.Tensor): Input frames with shape (frames, height, width, channels).
    coordinates (tuple): Center coordinates for cropping (center_y, center_x).
    target_dimension (tuple): Target dimensions (target_height, target_width).

    Returns:
    tf.Tensor: Cropped frames of shape (frames, target_height, target_width, channels).
    """
    input_height, input_width = frames.shape[1], frames.shape[2]

    # Extract center coordinates
    center_y, center_x = coordinates
    center_y = tf.cast(center_y, tf.int32)
    center_x = tf.cast(center_x, tf.int32)

    # Extract target dimensions
    target_height, target_width = target_dimension
    target_height = tf.cast(target_height, tf.int32)
    target_width = tf.cast(target_width, tf.int32)

    # Calculate cropping parameters
    y_start = tf.cast(tf.math.maximum(center_y - target_height // 2, 0), tf.int32)
    y_end = y_start + target_height
    x_start = tf.cast(tf.math.maximum(center_x - target_width // 2, 0), tf.int32)
    x_end = x_start + target_width

    # Adjust if the calculated end points are outside the image dimensions
    if y_end > input_height:
        y_start = input_height - target_height
        y_end = input_height
    if x_end > input_width:
        x_start = input_width - target_width
        x_end = input_width

    # Crop the frames
    cropped_frames = tf.image.crop_to_bounding_box(frames, y_start, x_start, target_height, target_width)
    return cropped_frames

def crop_video(input_fp, centers, target_dimension):
    """
    Crop all samples around the specified coordinates to the target dimensions. For each sample, it takes the current frame and the 8 frames before it. Each sample corresponds to one row in the centers tensor.

    Args:
    input_fp (str): Input mp4 video filepath.
    centers (tf.Tensor): Center coordinates for cropping of shape (batch_size, 3), with each row corresponding to (frame_number, x_coord, y_coord).
    target_dimension (tuple): Target dimensions (target_height, target_width).

    Returns:
    tf.Tensor: Cropped frames of shape (batch_size, frames=9, target_height, target_width, channels=3).
    """
    # Extract information from centers and target dimensions
    batch_size = centers.shape[0]
    target_height, target_width = target_dimension

    # Initialize a list to hold the final cropped frames for each sample
    final_cropped_frames = []

    # Iterate over each sample in the batch
    for i in range(batch_size):
        print(f"Processing {i}")
        if i == 10:
            break
        frame_number, center_x, center_y = centers[i].numpy()

        # Calculate the starting frame index for the 9 frames
        start_frame = max(frame_number - 8, 0)
        end_frame = frame_number + 1

        # Initialize the video reader and read the required frames
        video_reader = skvideo.io.vreader(input_fp)
        nine_frames = []
        for j, frame in enumerate(video_reader):
            if j >= start_frame and j < end_frame:
                nine_frames.append(frame)
            if j >= end_frame:
                break

        # If we didn't get exactly 9 frames (e.g., near the start of the video), pad with the first frame
        while len(nine_frames) < 9:
            print("This should never happen")
            nine_frames.insert(0, nine_frames[0])

        # Convert frames to tf.Tensor
        nine_frames = tf.convert_to_tensor(np.array(nine_frames), dtype=tf.float32)

        # Crop the frames using crop_single
        cropped_frames = crop_single(nine_frames, (center_y, center_x), (target_height, target_width))

        # Append the cropped frames to the final list
        final_cropped_frames.append(cropped_frames)

    # Stack all the cropped frames into a single tensor
    final_cropped_tensor = tf.stack(final_cropped_frames)
    return final_cropped_tensor

# # Example usage:
# input_fp = 'path_to_video/video_file.mp4'
# centers = tf.constant([[10, 50, 50], [20, 60, 60], [30, 70, 70]], dtype=tf.int32)  # Example centers tensor
# target_dimension = (100, 100)
# cropped_frames = crop_video(input_fp, centers, target_dimension)
# print(cropped_frames.shape)  # Should print (batch_size, frames=9, target_height, target_width, channels=3)
