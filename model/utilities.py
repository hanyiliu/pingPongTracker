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
    print(f"downscaled min max: {tf.reduce_min(downscaled_frames)}, {tf.reduce_max(downscaled_frames)}")

    return downscaled_frames

def crop_single(frames, coordinates, target_dimension):
    """
    Crop one sample's frames around the specified coordinates to the target dimensions.

    Args:
    frame (tf.Tensor): Input frame with shape (frames, height, width, channels).
    coordinates (tuple): Center coordinates for cropping (center_y, center_x).
    target_dimension (tuple): Target dimensions (target_height, target_width).

    Returns:
    tf.Tensor: Cropped frame of shape (frames, target_height, target_width, channels).
    """
    input_height = 1080
    input_width = 1920
    #print(f"Single frames: {frames.shape}, Coordinates: {coordinates}, Target: {target_dimension}")
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
        # y_end = tf.cast(y_end, tf.int64)  # Ensure consistent dtype
    if x_end > input_width:
        x_start = input_width - target_width
        x_end = input_width
        # x_end = tf.cast(x_end, tf.int64)  # Ensure consistent dtype

    # Crop the frames
    cropped_frames = tf.image.crop_to_bounding_box(
        frames, y_start, x_start, target_height, target_width)  # Corrected cropping range
    #print(f"cropped shape: {cropped_frames.shape}")
    return cropped_frames

def crop(frames, coordinates, target_dimension):
    """
    Crop all sample around the specified coordinates to the target dimensions.

    Args:
    frame (tf.Tensor): Input frame with shape (batch_size, frames, height, width, channels).
    coordinates (tuple of tf.Tensor): Center coordinates for cropping (center_y, center_x).
    target_dimension (tuple): Target dimensions (target_height, target_width).

    Returns:
    tf.Tensor: Cropped frame of shape (batch_size, frames, target_height, target_width, channels).
    """

    # Unpack dimensions
    batch_size, frames_count, _, _, channels_count = frames.shape
    target_height, target_width = target_dimension

    cropped_frames = tf.cast(tf.zeros((batch_size, frames_count, target_height, target_width, channels_count)), tf.float32)

    for sample_index in range(frames.shape[0]):
        sample = frames[sample_index]
        cropped_sample = crop_single(sample, (coordinates[0][sample_index], coordinates[1][sample_index]), target_dimension) # Return shape (frames_count, target_height, target_width, channels_count)
        #print(f"Processing sample {sample_index}: {sample.shape}")
        #print(f"sample_index type: {type(sample_index)}")
        indices = tf.constant([[sample_index]])
        update = tf.cast(tf.expand_dims(cropped_sample, 0), tf.float32)
        cropped_frames = tf.cast(cropped_frames, tf.float32)
        cropped_frames = tf.tensor_scatter_nd_update(cropped_frames, indices, update)


    #print(f"End of crop function: {cropped_frames.shape}")
    batch_size, frames_count, target_height, target_width, channels_count = cropped_frames.shape
    cropped_frames = tf.reshape(cropped_frames, (batch_size, frames_count, target_height, target_width * channels_count))
    cropped_frames = tf.transpose(cropped_frames, perm=[0, 2, 3, 1])  # shape: (batch_size, target_height, target_width * channels, frames)
    cropped_frames = tf.reshape(cropped_frames, (batch_size, target_height, target_width, frames_count * channels_count))  # shape: (batch_size, target_height, target_width, frames * channels)
    #print(f"Returning crop shape: {cropped_frames.shape}")

    return cropped_frames

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
