import matplotlib.pyplot as plt
import numpy as np
import json


def see_frames():
    # Example array of numbers


    with open("data/game_1_ball_markup.json", 'r') as file:
        data = json.load(file)

    frame_numbers = sorted([int(key) for key in data.keys()])
    numbers = np.array(frame_numbers)

    numbers = numbers[numbers < 60]
    print(numbers) #14-56 -> frame 22-56 -> 34 samples
    # Create a figure and a single subplot
    plt.figure(figsize=(12, 2))  # A wider figure to accommodate more points
    ax = plt.gca()

    # Plot each number on the number line
    plt.plot(numbers, np.zeros_like(numbers), 'ro', markersize=2)  # Smaller marker size for clarity

    # Customize the plot
    plt.ylim(-1, 1)  # Limit y-axis to show the markers clearly
    plt.yticks([])   # Hide y-axis ticks
    plt.xticks(np.linspace(0, 58, num=9))  # Set x-axis ticks to cover the range
    plt.grid(axis='x', linestyle='--')  # Add grid lines for better visibility

    # Add a title
    plt.title('Number Line for Large Array of Numbers')

    # Show the plot
    plt.show()

see_frames()
