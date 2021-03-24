import data_visualization
import pandas as pd
import matplotlib.pyplot as plt

key_pts_frame = pd.read_csv('./data/training_frames_keypoints.csv')

plt.figure(figsize=(10, 10))
data_visualization.show_keypoints(idx=44, key_pts_frame=key_pts_frame)
plt.show()