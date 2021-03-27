import os
from models.summary_model import get_summary_model
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Flatten, Dense, Dropout


def generate_new_model():
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(os.path.join(parent_dir, 'summary_model_weights'), 'vgg_face_weights.h5')
    summary_model = get_summary_model(model_path)
    x = Conv2D(512, kernel_size=(1, 1), padding='same')(summary_model.layers[-3].output)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(units=1024, use_bias=False)(x)
    x = Dropout(0.3)(x)
    x = Activation('relu')(x)
    x = Dense(512, use_bias=False)(x)
    x = Dropout(0.3)(x)
    x = Activation('relu')(x)
    x = Dense(136, activation='sigmoid')(x)
    model = Model(inputs=summary_model.input, outputs=x)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model

def load_latest_model(models):
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    latest_model = sorted(models, key=lambda x: x.split('_')[1], reverse=True)[0]
    model = keras.models.load_model(os.path.join(os.path.join(parent_dir, 'saved_models'), latest_model))
    return model

def get_model():
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    models = os.listdir(path=os.path.join(parent_dir, 'saved_models'))
    if len(models) > 0:
        model = load_latest_model(models)
    else:
        model = generate_new_model()
    return model
