import os
import time
import json
import warnings
import keras
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LeakyReLU
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, load_model
configs = json.loads(open(os.path.join(os.path.dirname(__file__), 'configs.json')).read())
warnings.filterwarnings("ignore")


def build_network(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("tanh"))

    start = time.time()
    model.compile(
        loss=configs['model']['loss_function'],
        optimizer=configs['model']['optimiser_function'])

    print("> Compilation Time : ", time.time() - start)
    return model


def build_cls_network(layers):
    model = Sequential()
    model.add(GRU(
        input_dim=layers[0],
        output_dim=layers[1],
        recurrent_dropout=0.5,
        return_sequences=True))
    # model.add(Dropout(0.2))

    model.add(GRU(
        layers[2],
        recurrent_dropout=0.5,
        return_sequences=False))
    model.add(Dropout(0.5))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    model.add(Dense(
        output_dim=layers[4]))
    model.add(Activation(activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    start = time.time()
    model.compile(
        loss=configs['model']['loss_function'],
        optimizer=configs['model']['optimiser_function'],
        metrics=['accuracy'])

    print("> Compilation Time : ", time.time() - start)
    return model


def load_network(filename):
    # Load the h5 saved model and weights
    if os.path.isfile(filename):
        return load_model(filename)
    else:
        print('ERROR: "' + filename + '" file does not exist as a h5 model')
        return None
