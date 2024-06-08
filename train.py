import config
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model.model import Model
from inputProcessing import video_to_tensor, temporary_format_input, format_input, format_output, format_output_bell_curve

input = video_to_tensor(config.input_fp)

start_frame = 22
end_frame = 56
batch_size = end_frame - start_frame
input = format_input(input, start_frame, end_frame)
# print(f"input: {input.shape}")
#print(f"Input: {input.shape}")
# Input shape: (batch_size, 9, 1080, 1920, 3)

frames = tf.range(start_frame, end_frame)
frames = tf.reshape(frames, (-1, 1))
# Create the tensor
frame_numbers = tf.constant(frames, dtype=tf.int32)

formatted_outputs = format_output("data/game_1_ball_markup.json")
# print(f"formatted_outputs: {formatted_outputs.shape}")
output = format_output_bell_curve(frame_numbers, formatted_outputs)

model = Model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model for 3 epochs
print(f"Input: {input.shape}")
print(f"Output: x: {output[0].shape}, y: {output[1].shape}")

model.fit(input, output, batch_size=batch_size, epochs=100)

predictions = model.predict(input, batch_size=batch_size)
print(f"Predictions: {predictions[0].shape} and {predictions[1].shape}")
x_guess = tf.argmax(predictions[0], axis=1)
y_guess = tf.argmax(predictions[1], axis=1)

x_actual = tf.argmax(output[0], axis=1)
y_actual = tf.argmax(output[1], axis=1)

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
