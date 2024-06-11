import cv2
import tensorflow as tf
import numpy as np
import json

from model.model import GlobalModel

class DataLoader:
    """
    A data loader class that loads a video file in batches and processes the frames.

    Args:
        video_file (str): The path to the video file.
        formatted_outputs (tf.Tensor): A matrix of the formatted output of shape (# of frames, 3), where each column corresponds to (frame number, x coord, y coord).
        batch_size (int): The number of frames to process in each batch.

    Yields:
        A tuple of two tensors. The first tensor is the input data of shape (batch_size, frames=9, height, width, channels=3) representing the processed frames.
        The second tensor is the output data of shape (batch_size, width) for x, and (batch_size, height) for y.
    """

    def __init__(self, video_file, formatted_outputs, batch_size):
        """
        Initializes the data loader.

        Args:
            video_file (str): The path to the video file.
            formatted_outputs (tf.Tensor): A matrix of the formatted output of shape (# of frames, 3), where each column corresponds to (frame number, x coord, y coord).
            batch_size (int): The number of frames to process in each batch.
        """
        self.video_file = video_file
        self.formatted_outputs = formatted_outputs
        self.batch_size = batch_size
        self.cap = cv2.VideoCapture(video_file)

    def __call__(self):
        return self.__iter__()

    def __iter__(self):
        frame_numbers = []
        for i in range(len(self.formatted_outputs)):
            frame_numbers.append(i)
        frame_numbers = tf.convert_to_tensor(frame_numbers, dtype=tf.int32)

        i = 0
        while True:
            frames = []
            for _ in range(self.batch_size):
                ret, frame = self.cap.read()
                if not ret:
                    break
                frames.append(frame)
            if not frames:
                break
            input_data = self.process_frames(frames)
            output_data = format_output_bell_curve(frame_numbers[i:i+self.batch_size], self.formatted_outputs)
            i += self.batch_size

            print("Input data shape:", input_data.shape)
            print("Output data shape:", output_data[0].shape, output_data[1].shape)

            yield input_data, output_data


    def process_frames(self, frames):
        """
        Processes a batch of frames to get a tensor of shape (batch_size, frames=9, height, width, channels=3).

        Args:
            frames (list): A list of frames to process.

        Returns:
            A tensor of shape (batch_size, frames=9, height, width, channels=3) representing the processed frames.
        """
        processed_frames = []
        for frame in frames:
            # Read the current frame and the 8 previous frames
            frames = []
            for _ in range(9):
                ret, frame = self.cap.read()
                if ret:
                    frames.append(frame)
                else:
                    break
            # If 9 frames are not available, pad with zeros
            while len(frames) < 9:
                frames.insert(0, np.zeros_like(frames[-1]))
            processed_frames.append(tf.convert_to_tensor(frames, dtype=tf.float32) / 255.0)
        return tf.stack(processed_frames)

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
        frame_number = frame_numbers[i]
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

# Create a data loader instance
formatted_outputs = format_output("data/game_1_ball_markup.json")
data_loader = DataLoader('data/downscaled_game_1.mp4', formatted_outputs, batch_size=32)

# Create a TensorFlow dataset from the data loader
dataset = tf.data.Dataset.from_generator(data_loader, output_types=(tf.float32, (tf.float32, tf.float32)))
print(dataset.element_spec)
for elem in dataset:
    print(elem)
# Batch the data into the desired batch size
dataset = dataset.batch(32)

# Prefetch the batches of data to improve performance
dataset = dataset.prefetch(tf.data.AUTOTUNE)


# Create a TensorFlow model
model = GlobalModel()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the dataset
model.fit(dataset, epochs=10)
