# Ping Pong Ball Tracker

A computer vision and deep learning project that uses TensorFlow to identify the location of a table tennis ball across sequential video frames.

## Overview

This project explores machine-learning-based ball tracking from table tennis video.

The model processes sequences of consecutive frames and learns to estimate the ball's horizontal and vertical position. Training data consists of recorded table tennis footage paired with frame-level ball-coordinate annotations.

The project uses a coarse-to-fine architecture with global and local localization stages.

## How It Works

Each training sample contains nine consecutive video frames:

- The current frame
- The previous eight frames

The frames are reformatted into tensors and processed by a convolutional neural network.

The model produces separate outputs for the ball's:

- X coordinate
- Y coordinate

Ground-truth coordinates are converted into bell-curve target distributions for training.

## Model Architecture

### Feature Extraction

The TensorFlow/Keras feature extractor includes:

- Convolutional layers
- Batch normalization
- ReLU activations
- Max pooling
- Dropout
- Fully connected layers

The convolutional stages increase feature depth from 64 to 128 to 256 filters.

### Coordinate Extraction

Separate neural-network heads estimate horizontal and vertical ball positions.

### Global Stage

The global model estimates the ball location from downscaled video frames.

Downscaling reduces computational cost while preserving enough spatial information for coarse localization.

### Local Stage

The local stage crops a smaller region around the ball location and performs more focused localization.

## Data Processing

The preprocessing pipeline:

1. Reads table tennis video files
2. Extracts selected video frames
3. Converts frames into TensorFlow tensors
4. Loads manually labeled ball coordinates from JSON
5. Creates sequences of nine consecutive frames
6. Downscales and reformats input tensors
7. Converts coordinate labels into x/y target distributions

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-video
- Computer Vision
- Deep Learning

## Repository Structure

```text
.
├── data/
│   ├── downscaled_game_1.mp4
│   └── game_1_ball_markup.json
├── model/
│   ├── coordExtractor.py
│   ├── crop.py
│   ├── featureExtractor.py
│   ├── model.py
│   └── utilities.py
├── convert_to_tensor.py
├── dataloader.py
├── global_train.py
├── inputProcessing.py
├── local_train.py
└── README.md
