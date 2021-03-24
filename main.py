from data_preprocessing import get_ds, process_ds
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

keys_frame_train = pd.read_csv('./data/training_frames_keypoints.csv')
keys_frame_test = pd.read_csv('./data/test_frames_keypoints.csv')

train_ds = get_ds(keys_frame_train)
test_ds = get_ds(keys_frame_test)

train_ds = train_ds.map(process_ds(ds_type='training', new_shape=(224, 224, 3)), num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(process_ds(ds_type='testing', new_shape=(224, 224, 3)), num_parallel_calls=tf.data.AUTOTUNE)

for i, k in iter(train_ds):
    k = k.numpy() * 224.0
    k = k.reshape(-1, 2)
    plt.imshow(i)
    plt.scatter(k[:, 0], k[:, 1], s=20, c='m', marker='.')
    plt.show()
    break
