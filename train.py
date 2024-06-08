import config
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math

from model.model import Model, GlobalModel
from model.utilities import downscale
from inputProcessing import video_to_tensor, temporary_format_input, format_input, format_output, format_output_bell_curve

def visualize_first_frame_first_batch_sample(tensor):
    # Ensure the tensor has the expected shape
    if tensor.shape[-1] != 3:
        raise ValueError("The number of channels is not 3.")

    # Select the first batch sample and the first frame
    first_frame = tensor[0, 0, :, :, :].numpy()

    # Display the frame
    plt.figure(figsize=(6, 6))
    plt.imshow(first_frame)
    plt.axis("off")
    plt.title("First Frame of the First Batch Sample")
    plt.show()


# Example usage
# visualize_images(downscaled_input)


input = video_to_tensor(config.input_fp)
print(f"input shape: {input.shape}")
start_frame = 22
end_frame = 56
batch_size = end_frame - start_frame

input = format_input(input, start_frame, end_frame)
print(f"formatted input shape: {input.shape}")
downscaled_input = downscale(input, (128, 320))
downscaled_input = tf.cast(downscaled_input, tf.int32)

visualize_first_frame_first_batch_sample(downscaled_input)

frames = tf.range(start_frame, end_frame)
frames = tf.reshape(frames, (-1, 1))
# Create the tensor
frame_numbers = tf.constant(frames, dtype=tf.int32)

formatted_outputs = format_output("data/game_1_ball_markup.json")


downscaled_formatted_outputs = tf.cast(formatted_outputs, tf.float32)
downscaled_frame_numbers = downscaled_formatted_outputs[:, 0]
downscaled_x_outputs = downscaled_formatted_outputs[:, 1] * (320 / 1920)
downscaled_y_outputs = downscaled_formatted_outputs[:, 2] * (128 / 1080)

downscaled_formatted_outputs = tf.stack([downscaled_frame_numbers, downscaled_x_outputs, downscaled_y_outputs], axis=1)
downscaled_formatted_outputs = tf.cast(downscaled_formatted_outputs, tf.int32)

# print(f"formatted_outputs: {formatted_outputs.shape}")
downscaled_output = format_output_bell_curve(frame_numbers, downscaled_formatted_outputs, width=320, height=128)
output = format_output_bell_curve(frame_numbers, formatted_outputs)

# model = Model()
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

global_model = GlobalModel()
print(f"downscaled_input shape: {downscaled_input.shape}")
global_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
global_model.fit(downscaled_input, downscaled_output, batch_size=batch_size, epochs=3)


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

# # Convert TensorFlow tensor to NumPy array
# numpy_data = predictions[0]
# # Plotting the data
# # Create a figure with 5 subplots, one for each row
# fig, axs = plt.subplots(batch_size, 1, figsize=(12, 20), sharex=True)
#
# # Plot each row in a separate subplot
# for i in range(numpy_data.shape[0]):
#     axs[i].plot(numpy_data[i])
#     axs[i].set_title(f'Series {i+1}')
#     axs[i].set_ylabel('Value')
#
# # Set the xlabel for the last subplot
# axs[-1].set_xlabel('Index')
#
# # Adjust layout
# plt.tight_layout()
#
# # Show the plot
# plt.show()
# model.build((5, 9, 1080, 1920, 3))

# print(model.summary())
