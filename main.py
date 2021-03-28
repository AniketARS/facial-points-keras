import pandas as pd
from data_preprocessing import get_ds, process_ds
import tensorflow as tf
from model_training import train_model


def representative_dataset():
    sample_ds = get_ds(pts_frame_train)
    sample_ds = sample_ds.map(process_ds(ds_type='training', new_shape=(224, 224, 3)),
                              num_parallel_calls=tf.data.AUTOTUNE)
    sample_ds = sample_ds.batch(1).take(200)
    for data in sample_ds:
        yield [data[0]]


if __name__ == '__main__':

    pts_frame_train = pd.read_csv('./data/training_frames_keypoints.csv')
    pts_frame_test = pd.read_csv('./data/test_frames_keypoints.csv')

    train_ds = get_ds(pts_frame_train)
    test_ds = get_ds(pts_frame_test)

    train_ds = train_ds.map(process_ds(ds_type='training', new_shape=(224, 224, 3)), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(5000).batch(32).prefetch(1)
    test_ds = test_ds.map(process_ds(ds_type='testing', new_shape=(224, 224, 3)), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(32).prefetch(1)

    history = train_model(train_ds, test_ds, sample_ds=representative_dataset, save_converted=True)