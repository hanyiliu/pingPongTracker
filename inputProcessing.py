import numpy
numpy.float = numpy.float64
numpy.int = numpy.int_
import json

import tensorflow as tf
numpy.set_printoptions(threshold=2000)

import skvideo.io # For mp4 to tensors processing
import config
import matplotlib.pyplot as plt

def save_tensor(tensor, file_path):
    """
    Save a tensor to a local file.

    Args:
    tensor (tf.Tensor): The tensor to save.
    file_path (str): The file path where the tensor will be saved.
    """
    serialized_tensor = tf.io.serialize_tensor(tensor)
    tf.io.write_file(file_path, serialized_tensor)

def load_tensor(file_path):
    """
    Load a tensor from a local file.

    Args:
    file_path (str): The file path from which to load the tensor.

    Returns:
    tf.Tensor: The loaded tensor.
    """
    serialized_tensor = tf.io.read_file(file_path)
    tensor = tf.io.parse_tensor(serialized_tensor, out_type=tf.float32)
    return tensor

def video_to_tensor(video_fp, frame_indices_tensor):
    """
    Load a video and extract specific frames.

    Args:
    video_fp (str): File path to the video.
    frame_indices_tensor (tf.Tensor): Tensor containing the frame indices to extract.

    Returns:
    tf.Tensor: Tensor containing the extracted frames.
    """
    # Convert frame_indices_tensor to a numpy array
    frame_indices = frame_indices_tensor.numpy()

    # Initialize the video reader
    video_reader = skvideo.io.vreader(video_fp)

    # Initialize a list to hold the extracted frames
    frames_list = []

    # Iterate over the video frames and extract the desired frames
    for i, frame in enumerate(video_reader):
        #print(f"i: {i}")
        if i in frame_indices:
            print(f"{i} Found")
            frames_list.append(frame)

    # Convert the list of frames to a tensor
    video_tensor = tf.convert_to_tensor(np.array(frames_list))
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

def format_input(formatted_outputs, video_tensor):
    """
    Create a formatted input tensor with shape (batch_size, frames=9, height, width, 3)
    from the given formatted outputs and video tensor.

    Args:
    formatted_outputs (tf.Tensor): Tensor with shape (batch_size, 3), where each row corresponds to (frame_number, x_coord, y_coord).
    video_tensor (tf.Tensor): Tensor with shape (# of frames, height, width, channels=3).

    Returns:
    tf.Tensor: Formatted input tensor with shape (batch_size, frames=9, height, width, 3).
    """
    # Get the shape of the video tensor
    num_frames, height, width, channels = video_tensor.shape

    # Initialize a list to hold the formatted input for each sample
    formatted_input_list = []

    for frame_info in formatted_outputs:
        frame_number = frame_info[0].numpy()

        # Ensure we have at least 9 frames including the current one
        if frame_number < 8:
            raise ValueError(f"Frame number {frame_number} is too small to get 9 frames including the current one.")

        # Extract the current frame and its previous 8 frames
        frames = video_tensor[frame_number-8:frame_number+1]

        # Append to the list
        formatted_input_list.append(frames)

    # Stack the list into a single tensor
    formatted_input = tf.stack(formatted_input_list)

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
