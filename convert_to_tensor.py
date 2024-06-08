import config
import tensorflow as tf


from inputProcessing import video_to_tensor, save_tensor


input = video_to_tensor("data/game_1.mp4")
save_tensor(input, "data/game_1")
