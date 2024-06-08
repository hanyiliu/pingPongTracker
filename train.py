import config
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model.model import Model, GlobalModel
from model.utilities import downscale
from inputProcessing import video_to_tensor, temporary_format_input, format_input, format_output, format_output_bell_curve

start_frame = 22
end_frame = 56

input = video_to_tensor(config.input_fp)
input = format_input(input, start_frame, end_frame)
print(f"input shape: {input.shape}")

formatted_outputs = format_output("data/game_1_ball_markup.json")

frames = tf.range(start_frame, end_frame)
frames = tf.reshape(frames, (-1, 1))
frame_numbers = tf.constant(frames, dtype=tf.int32)

output = format_output_bell_curve(frame_numbers, formatted_outputs)
x_output = tf.argmax(output[0], axis=1)
y_output = tf.argmax(output[1], axis=1)
print(f"outputs shape: {x_output.shape}, {y_output.shape}")

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

def crop(frames, coordinates, target_dimension):
    """
    Crop all samples around the specified coordinates to the target dimensions.

    Args:
    frames (tf.Tensor): Input frames with shape (batch_size, frames, height, width, channels).
    coordinates (tuple of tf.Tensor): Center coordinates for cropping (center_y, center_x).
    target_dimension (tuple): Target dimensions (target_height, target_width).

    Returns:
    tf.Tensor: Cropped frames of shape (batch_size, frames, target_height, target_width, channels).
    """

    # Unpack dimensions
    batch_size, frames_count, _, _, channels_count = frames.shape
    target_height, target_width = target_dimension

    cropped_frames = tf.TensorArray(tf.float32, size=batch_size)

    for sample_index in range(batch_size):
        sample = frames[sample_index]
        cropped_sample = crop_single(sample, (coordinates[0][sample_index], coordinates[1][sample_index]), target_dimension)
        cropped_sample = tf.cast(cropped_sample, tf.float32)  # Ensure dtype is float32
        cropped_frames = cropped_frames.write(sample_index, cropped_sample)

    cropped_frames = cropped_frames.stack()
    return cropped_frames

cropped = crop (input, (y_output,x_output),(128,320))

print(f"cropped shape: {cropped.shape}")

def display_frames(original_frames, cropped_frames):
    """
    Display original and cropped frames using Matplotlib.

    Args:
    original_frames (tf.Tensor): Tensor of shape (batch_size, frames, height, width, channels).
    cropped_frames (tf.Tensor): Tensor of shape (batch_size, frames, target_height, target_width, channels).
    """
    num_frames = original_frames.shape[1]
    batch_size = original_frames.shape[0]

    for b in range(batch_size):
        fig, axes = plt.subplots(2, num_frames, figsize=(20, 8))

        for i in range(num_frames):
            # Plot original frame
            ax_orig = axes[0, i]
            ax_orig.imshow(original_frames[b, i].numpy().astype(np.uint8))
            ax_orig.set_title(f'Original Frame {i+1}')
            ax_orig.axis('off')

            # Plot cropped frame
            ax_cropped = axes[1, i]
            ax_cropped.imshow(cropped_frames[b, i].numpy().astype(np.uint8))
            ax_cropped.set_title(f'Cropped Frame {i+1}')
            ax_cropped.axis('off')

        plt.show()

# Assuming frames_tensor and cropped_frames are already obtained from your crop function
# For example:
# frames_tensor = tf.convert_to_tensor(your_original_frames)
# cropped_frames = crop(frames_tensor, (y_output, x_output), target_dimension)

# Display the frames
display_frames(input, cropped)
