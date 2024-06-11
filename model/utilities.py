import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

# Input size: 3x1080x1920
# Output size: 3x128x320

def downscale(frames, target_dimension):
    """
    Downscale all sample's frames from the original dimensions to the target dimensions.

    Args:
    frames (tf.Tensor): Input frame with shape (batch_size, frames, original_height, original_width, channels).
    target_dimension (tuple): Target dimensions (target_height, target_width).

    Returns:
    tf.Tensor: Downscaled frames of shape (batch_size, frames, target_height, target_width, channels).
    """
    target_height, target_width = target_dimension

    # Get the batch size and number of frames
    batch_size, num_frames, original_height, original_width, channels = frames.shape

    # Reshape to merge batch_size and num_frames to apply resize function
    reshaped_frames_before_resize = tf.reshape(frames, (-1, original_height, original_width, channels))
    print(f"Reshaped before resizing: {reshaped_frames_before_resize.shape}")

    # Resize the frames
    resized_frames = tf.image.resize(reshaped_frames_before_resize, [target_height, target_width], method=tf.image.ResizeMethod.BILINEAR)

    # Reshape back to the original structure
    downscaled_frames = tf.reshape(resized_frames, (batch_size, num_frames, target_height, target_width, channels))
    print(f"Reshaped after resizing: {downscaled_frames.shape}")
    #print(downscaled_frames)

    return downscaled_frames
#
# def crop_single(frames, coordinates, target_dimension):
#     """
#     Crop one sample's frames around the specified coordinates to the target dimensions.
#
#     Args:
#     frames (tf.Tensor): Input frames with shape (frames, height, width, channels).
#     coordinates (tuple): Center coordinates for cropping (center_y, center_x).
#     target_dimension (tuple): Target dimensions (target_height, target_width).
#
#     Returns:
#     tf.Tensor: Cropped frames of shape (frames, target_height, target_width, channels).
#     """
#     input_height, input_width = frames.shape[1], frames.shape[2]
#
#     # Extract center coordinates
#     center_y, center_x = coordinates
#     center_y = tf.cast(center_y, tf.int32)
#     center_x = tf.cast(center_x, tf.int32)
#
#     # Extract target dimensions
#     target_height, target_width = target_dimension
#     target_height = tf.cast(target_height, tf.int32)
#     target_width = tf.cast(target_width, tf.int32)
#
#     # Calculate cropping parameters
#     y_start = tf.cast(tf.math.maximum(center_y - target_height // 2, 0), tf.int32)
#     y_end = y_start + target_height
#     x_start = tf.cast(tf.math.maximum(center_x - target_width // 2, 0), tf.int32)
#     x_end = x_start + target_width
#
#     # Adjust if the calculated end points are outside the image dimensions
#     if y_end > input_height:
#         y_start = input_height - target_height
#         y_end = input_height
#     if x_end > input_width:
#         x_start = input_width - target_width
#         x_end = input_width
#
#     # Crop the frames
#     cropped_frames = tf.image.crop_to_bounding_box(frames, y_start, x_start, target_height, target_width)
#     return cropped_frames
#

# Crop in the past:
# def crop(frames, coordinates, target_dimension):
    # # Unpack dimensions
    # batch_size, frames_count, _, _, channels_count = frames.shape
    # target_height, target_width = target_dimension
    #
    # cropped_frames = tf.TensorArray(tf.float32, size=batch_size)
    #
    # for sample_index in range(batch_size):
    #     sample = frames[sample_index]
    #     cropped_sample = crop_single(sample, (coordinates[0][sample_index], coordinates[1][sample_index]), target_dimension)
    #     cropped_sample = tf.cast(cropped_sample, tf.float32)  # Ensure dtype is float32
    #     cropped_frames = cropped_frames.write(sample_index, cropped_sample)
    #
    # cropped_frames = cropped_frames.stack()
    # return cropped_frames

def crop_output(output, center, target_length):
    """
    Crop output with center at center and having a target length of target_length.

    Args:
    output (tf.Tensor): Tensor of shape (batch_size, original_length).
    center (tf.Tensor): Tensor of shape (batch_size, 1), where each element is the center of the desired crop
    target_length (int): Target length

    Returns:
    tf.Tensor: Cropped tensor of shape (batch_size, target_length).
    """

    batch_size = tf.shape(output)[0]
    original_length = tf.shape(output)[1]

    # Calculate start and end indices for cropping
    half_length = target_length // 2
    start_indices = tf.maximum(center - half_length, 0)
    end_indices = tf.minimum(center + half_length + target_length % 2, original_length)

    # Ensure the indices are within the valid range
    start_indices = tf.minimum(start_indices, original_length - target_length)
    end_indices = tf.maximum(end_indices, target_length)

    # Create a range of indices to gather
    indices = tf.range(target_length)

    # Gather the cropped segments
    cropped_output = tf.map_fn(
        lambda i: tf.gather(output[i], tf.range(start_indices[i], end_indices[i])),
        tf.range(batch_size),
        dtype=tf.float32
    )
    print(f"cropped output: {cropped_output.shape}")

    return cropped_output

def pad(cropped_samples, length0, length1, center1):
    """
    Pad a tensor of shape (batch_size, length0) with zeros into a new tensor of shape (batch_size, original_length).
    NOTE: each row's left and right padding will differ row by row.
    We find the pad on the left (pre_pad) and the pad on the right (post_pad) by using this algorithm:
        pre_pad = center1*(length0/length1) - (length1)/2
        post_pad = length0 - center1*(length0/length1) - (length1)/2

    Args:
    cropped_samples (tf.Tensor): a tensor of shape (batch_size, length0).
    length0 (int32): the length of the original dimension
    length1 (int32): the length of the cropped dimension
    center1 (tf.Tensor): a tensor of shape (batch_size, 1), where each value corresponds to the center that the cropped dimension's is cropped from.

    Returns:
    tf.Tensor: Padded tensor of shape (batch_size, original_length).
    """
    batch_size = cropped_samples.shape[0]

    #print(f"center1: {center1}")
    #print(f"length0, 1: {length0, length1}") # 1920, 320; 1080, 128

    # Calculate the left (pre) and right (post) padding for each row

    pre_pad = center1 - (length1 // 2)
    pre_pad = tf.maximum(pre_pad, 0)
    pre_pad = tf.minimum(length0 - length1, pre_pad)


    #print(f"pre pad: {pre_pad.shape}")
    padded_samples = tf.zeros((batch_size, length0))
    print(f"padded_sample shape: {padded_samples.shape}")
    print(f"batchsize: {batch_size}, length0: {length0}")
    for sample_index in range(0, batch_size):
        current_pre_pad = pre_pad[sample_index]
        current_post_pad = length0 - length1 - current_pre_pad
        #print(f"current_pre_pad: {current_pre_pad}")
        #print(f"current_post_pad: {current_post_pad}")

        current_pad = [current_pre_pad, current_post_pad]
        current_pad = tf.stack(current_pad, axis=0)

        current_sample = cropped_samples[sample_index]
        #print(f"current_sample: {current_sample.shape}")



        #print(f"current_pad: {current_pad.shape}")
        #print(f"$$$$$$$$ current_pad value: {current_pad}")

        current_padded_sample = tf.pad(current_sample, [current_pad], mode="CONSTANT", constant_values=0)
        #print(f"current_padded_sample: {current_padded_sample.shape}")
        #print(f"i type: {sample_index}")
        indices = tf.constant([[sample_index]])
        update = tf.cast(tf.expand_dims(current_padded_sample, 0), tf.float32)
        padded_samples = tf.tensor_scatter_nd_update(padded_samples, indices, update)

    #print(f"padded_samples: {padded_samples.shape}")

    return padded_samples
