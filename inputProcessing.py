import numpy
numpy.float = numpy.float64
numpy.int = numpy.int_
import json

import tensorflow as tf
numpy.set_printoptions(threshold=2000)

import skvideo.io # For mp4 to tensors processing
import config
import matplotlib.pyplot as plt

def video_to_tensor(video_fp):
    video_tensor = skvideo.io.vread(video_fp)
    print(video_tensor.shape)
    return video_tensor

def temporary_format_input(frames):
    """
    Format the input to our model's desired shapes.

    Args:
    frame (tf.Tensor): Input frame with shape (frames, height, width, channels).

    Returns:
    tf.Tensor: Reformatted training samples of input data of shape (batch_size, frames, target_height, target_width, channels).
    """

    frames_count, height, width, channels = frames.shape

    n_prev = 8

    batch_size = frames_count // (n_prev+1)

    formatted_input = []

    for i in range(batch_size):
        sample = frames[(n_prev + 1) * i:(n_prev + 1) * (i + 1)]
        ##print(f"i: {i}, Sample shape: {sample.shape}")
        formatted_input.append(tf.expand_dims(sample, axis=0))

    # Concatenate all the samples along the batch axis
    formatted_input = tf.concat(formatted_input, axis=0)
    #print(f"Final input: {formatted_input.shape}")
    return formatted_input

def format_input(frames, start_frame, end_frame):
    """
    Format the input to our model's desired shapes.

    Args:
    frame (tf.Tensor): Input frame with shape (frames, height, width, channels).
    start_frame (int): The frame to start formatting from.
    end_frame (int): The frame to stop formatting at.

    Returns:
    tf.Tensor: Reformatted training samples of input data of shape (batch_size, frames, target_height, target_width, channels).
    """
    frames_count, height, width, channels = frames.shape

    n_prev = 8

    batch_size = end_frame - start_frame

    formatted_input = []

    for i in range(batch_size):
        current_frame = start_frame + i
        sample = frames[current_frame - 8:current_frame + 1]
        ##print(f"i: {i}, Sample shape: {sample.shape}")
        formatted_input.append(tf.expand_dims(sample, axis=0))

    # Concatenate all the samples along the batch axis
    formatted_input = tf.concat(formatted_input, axis=0)
    #print(f"Final input: {formatted_input.shape}")
    return formatted_input

def format_output(output_fp):
    """
    Format the output to our model's desired values.

    Args:
    output_fp (string): Filepath to the JSON output values.

    Returns:
    tf.Tensor: Formatted tensor with shape (# of frames, 3).
    """
    with open(output_fp, 'r') as file:
        data = json.load(file)

    frame_numbers = sorted([int(key) for key in data.keys()])
    #print(f"frames: {frame_numbers}")
    num_frames = len(frame_numbers)
    output_tensor = tf.zeros((num_frames, 3))

    # Create indices to update in the tensor
    indices = tf.expand_dims(tf.range(num_frames), axis=1)
    updates = tf.constant([[frame_num, data[str(frame_num)]['x'], data[str(frame_num)]['y']] for frame_num in frame_numbers])

    # Cast updates tensor to float
    updates = tf.cast(updates, dtype=tf.float32)

    # Update the tensor using tf.tensor_scatter_nd_update
    output_tensor = tf.tensor_scatter_nd_update(output_tensor, indices, updates)
    output_tensor = tf.cast(output_tensor, dtype=tf.int32)

    return output_tensor

def format_output_bell_curve(frame_numbers, formatted_outputs, width=1920, height=1080, ball_radius=1):
    """
    Create bell curve vectors for the output.

    Args:
    frame_numbers (tf.Tensor): A vector of the desired frame numbers of shape (batch_size, 1).
    formatted_outputs (tf.Tensor): A matrix of the formatted output of shape (# of frames, 3), where each column corresponds to (frame number, x coord, y coord).

    Returns:
    tuple of tf.Tensor: Two formatted output tensors with shape (batch_size, width) for x, and (batch_size, height) for y.
    """

    batch_size = frame_numbers.shape[0]

    x_bell_curves = []
    y_bell_curves = []

    for i in range(batch_size):
        frame_number = frame_numbers[i, 0]
        frame_output = formatted_outputs[formatted_outputs[:, 0] == frame_number]

        if frame_output.shape[0] == 0:
            raise ValueError(f"Frame number {frame_number} not found in formatted_outputs.")

        _, x_coord, y_coord = frame_output[0]
        variance = ball_radius**2

        # Ensure x_coord and y_coord are float tensors
        x_coord = tf.cast(x_coord, tf.float32)
        y_coord = tf.cast(y_coord, tf.float32)

        # Create bell curve for x coordinate using TensorFlow operations
        x = tf.range(width, dtype=tf.float32)
        x_bell_curve = tf.exp(-tf.square(x - x_coord) / (2 * variance))

        # Create bell curve for y coordinate using TensorFlow operations
        y = tf.range(height, dtype=tf.float32)
        y_bell_curve = tf.exp(-tf.square(y - y_coord) / (2 * variance))

        x_bell_curves.append(x_bell_curve)
        y_bell_curves.append(y_bell_curve)

    x_bell_curves = tf.stack(x_bell_curves)
    y_bell_curves = tf.stack(y_bell_curves)

    return x_bell_curves, y_bell_curves
