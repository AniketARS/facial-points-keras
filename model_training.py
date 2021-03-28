import tensorflow.keras as keras
from models.model import get_model
import tensorflow as tf


def train_model(train_ds, test_ds, sample_ds, save_converted=True):
    epochs = 1
    early_stopper = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5,
                                                  restore_best_weights=True)
    model = get_model()
    print(model.summary())
    history = model.fit(train_ds, epochs=epochs, callbacks=[early_stopper], validation_data=test_ds)
    model.save('./saved_models/model_{}_epochs.h5'.format(epochs))
    if save_converted:
        save_model_converted(model, sample_ds)
    return history

def save_model_converted(model, sample_ds):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = sample_ds
    tflite_model = converter.convert()
    with open('./converted_model/model.tflite', 'wb') as f:
        f.write(tflite_model)
