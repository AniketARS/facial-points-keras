from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Activation


def get_summary_model(weights_path):
    summary_model = Sequential()
    summary_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
    summary_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    summary_model.add(MaxPool2D(pool_size=(2, 2)))
    summary_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    summary_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    summary_model.add(MaxPool2D(pool_size=(2, 2)))
    summary_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    summary_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    summary_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    summary_model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    summary_model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    summary_model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    summary_model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    summary_model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    summary_model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    summary_model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    summary_model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    summary_model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    summary_model.add(Conv2D(4096, (7, 7), activation='relu', padding='same'))
    summary_model.add(Dropout(0.5))
    summary_model.add(Conv2D(4096, (1, 1), activation='relu', padding='same'))
    summary_model.add(Dropout(0.5))
    summary_model.add(Conv2D(2622, (1, 1)))
    summary_model.add(Flatten())
    summary_model.add(Activation('softmax'))

    summary_model.load_weights(weights_path)
    for layer in summary_model.layers:
        layer.trainable = False

    return summary_model
