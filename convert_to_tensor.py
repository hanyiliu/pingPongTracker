import config
import tensorflow as tf


from inputProcessing import video_to_tensor, save_tensor, format_output

formatted_outputs = format_output("data/game_1_ball_markup.json")

input = video_to_tensor("data/game_1.mp4", formatted_outputs[:,0])
save_tensor(input, "data/game_1")
