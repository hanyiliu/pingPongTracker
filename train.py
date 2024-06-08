import config
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math

from model.model import Model, GlobalModel
from model.utilities import downscale
from inputProcessing import video_to_tensor, temporary_format_input, format_input, format_output, format_output_bell_curve, load_tensor

def visualize_all_frame(collapsed_frames, num_frames):
    """
    Visualize the first frame of the first sample from the collapsed frames tensor.

    Args:
    collapsed_frames (tf.Tensor): Input tensor with shape (batch_size, height, width, all channels of each frame).
    """
    collapsed_frames = tf.cast(collapsed_frames, tf.int32)

    # Define the number of channels per frame
    channels_per_frame = 3

    # Get the batch size (number of samples)
    batch_size = collapsed_frames.shape[0]

    # Iterate through each sample
    for sample_idx in range(batch_size):
        # Calculate the number of rows and columns for the grid
        rows = int(num_frames ** 0.5)
        cols = (num_frames + rows - 1) // rows

        # Create a figure with subplots for the current sample
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        axes = axes.flatten()

        # Iterate through each frame and visualize it
        for frame_idx in range(num_frames):
            start_channel = frame_idx * channels_per_frame
            end_channel = start_channel + channels_per_frame

            # Select the current frame for the current sample
            current_frame = collapsed_frames[sample_idx, :, :, start_channel:end_channel]

            # Display the frame
            axes[frame_idx].imshow(current_frame)
            axes[frame_idx].axis("off")
            axes[frame_idx].set_title(f"Sample {sample_idx + 1}, Frame {frame_idx + 1}")

        # Hide any unused subplots
        for j in range(frame_idx + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()


def collapse_channels(frames):
    """
    Collapse the tensor of frames to merge all channels of each frame into one channel dimension,
    and reorder the channels such that all red channels come first, followed by all green channels,
    and finally all blue channels.

    Args:
    frames (tf.Tensor): Input frames with shape (batch_size, frames, height, width, channels).

    Returns:
    tf.Tensor: Collapsed frames with shape (batch_size, height, width, frames * channels).
    """
    # Get the shape of the input tensor
    batch_size, num_frames, height, width, channels = frames.shape

    # Transpose the tensor to bring the channels to the front
    frames_transposed = tf.transpose(frames, perm=[0, 2, 3, 1, 4])

    # Reshape to merge the frames and channels dimensions
    collapsed_frames = tf.reshape(frames_transposed, (batch_size, height, width, num_frames * channels))

    return collapsed_frames

### OUTPUT ####
formatted_outputs = format_output("data/game_1_ball_markup.json")
print(f"formatted outputs shape: {formatted_outputs.shape}")

downscaled_formatted_outputs = tf.cast(formatted_outputs, tf.float32)
downscaled_frame_numbers = downscaled_formatted_outputs[:, 0]
downscaled_x_outputs = downscaled_formatted_outputs[:, 1] * (320 / 1920)
downscaled_y_outputs = downscaled_formatted_outputs[:, 2] * (128 / 1080)

downscaled_formatted_outputs = tf.stack([downscaled_frame_numbers, downscaled_x_outputs, downscaled_y_outputs], axis=1)
downscaled_formatted_outputs = tf.cast(downscaled_formatted_outputs, tf.int32)

# print(f"formatted_outputs: {formatted_outputs.shape}")
downscaled_output = format_output_bell_curve(tf.reshape(downscaled_frame_numbers, (-1, 1)), downscaled_formatted_outputs, width=320, height=128)
output = format_output_bell_curve(tf.reshape(formatted_outputs[:, 0], (-1, 1)), formatted_outputs)

#######

### INPUT ######

video_tensor = load_tensor("data/game_1")
print(f"input shape: {input.shape}")
start_frame = 22
end_frame = 56
batch_size = end_frame - start_frame

input = format_input(formatted_outputs, video_tensor)

downscaled_input = downscale(input, (128, 320))
downscaled_input = collapse_channels(downscaled_input)

print(f"downscaled_input shape: {downscaled_input.shape}")

print(f"downscaled_output shape: {downscaled_output[0].shape}, {downscaled_output[1].shape}")

global_model = GlobalModel()

global_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
global_model.fit(downscaled_input, downscaled_output, batch_size=batch_size, epochs=3)

# Save the model to a specified directory
global_model.save('global_model')



global_predictions = global_model.predict(downscaled_input, batch_size=batch_size)
print(f"Predictions: {global_predictions[0].shape} and {global_predictions[1].shape}")
x_guess = tf.argmax(global_predictions[0], axis=1)
y_guess = tf.argmax(global_predictions[1], axis=1)

x_actual = tf.argmax(downscaled_output[0], axis=1)
y_actual = tf.argmax(downscaled_output[1], axis=1)

print(f"X_guess: {x_guess}")
print(f"Y_guess: {y_guess}")

print(f"Actual X: {x_actual}")
print(f"Actual Y: {y_actual}")
