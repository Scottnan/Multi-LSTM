import os
import time
import lstm, etl, json
import numpy as np
import plot_utils as plot
import keras
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import h5py
from keras.callbacks import TensorBoard
configs = json.loads(open('configs.json').read())
tstart = time.time()
log_time = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
# TODO plot learning curve and clean the code


class MulLSTM(object):
    def __init__(self, configs):
        self.method = configs['model']['method']
        self.model_path = configs['model']['filename_model']
        self.raw_path = configs['data']['filename']
        self.clean_path = configs['data']['filename_clean']
        self.batch_size = configs['data']['batch_size']
        self.x_window_size = configs['data']['x_window_size'],
        self.y_window_size = configs['data']['y_window_size'],
        self.y_col = configs['data']['y_predict_column'],  # TODO change y_col to y_lag, and y_col is useless now.
        self.filter_cols = configs['data']['filter_columns']
        self.train_test_split = configs['data']['train_test_split']
        self.model = None
        self.dl = etl.ETL(self.method)

    def clean_data(self):
        if os.path.exists(self.clean_path):
            print('> Clean data exists!')
        else:
            self.dl.create_clean_datafile(
                filename_in=self.raw_path,
                filename_out=self.clean_path,
                batch_size=self.batch_size,
                x_window_size=self.x_window_size,
                y_window_size=self.y_window_size,
                y_col=self.y_col,  # TODO change y_col to y_lag, and y_col is useless now.
                filter_cols=self.filter_cols,
                normalise=False
            )
            print('> Generating clean data from:', self.clean_path, 'with batch_size:', self.batch_size)

    def fit(self):
        with h5py.File(self.clean_path, 'r') as hf:
            nrows = hf['x'].shape[0]
            ncols = hf['x'].shape[2]

        ntrain = int(self.train_test_split * nrows)
        steps_per_epoch = int(ntrain / self.batch_size)
        trsize = steps_per_epoch * self.batch_size

        with h5py.File(self.clean_path, 'r') as hf:
            val_data_x = hf['x'][ntrain:]
            val_data_y = hf['y'][ntrain:]
            if dl.method == 'Integer' or dl.method == "OneHot":
                val_data_y = keras.utils.to_categorical(val_data_y - 1, num_classes=3)

        val_data = (val_data_x, val_data_y)
        data_gen_train = dl.generate_clean_data(
            configs['data']['filename_clean'],
            size=trsize,
            batch_size=configs['data']['batch_size']
        )

        print('> Clean data has', nrows, 'data rows. Training on', ntrain, 'rows with', steps_per_epoch,
              'steps-per-epoch')

        self.model = lstm.build_cls_network([ncols, 400, 400, 100])
        self.fit_model(data_gen_train, steps_per_epoch, configs, val_data)

    def predict(self):
        ntest = nrows - ntrain
        steps_test = int(ntest / configs['data']['batch_size'])
        tesize = steps_test * configs['data']['batch_size']

        data_gen_test = dl.generate_clean_data(
            configs['data']['filename_clean'],
            size=tesize,
            batch_size=configs['data']['batch_size'],
            start_index=ntrain
        )

        print('> Testing model on', ntest, 'data rows with', steps_test, 'steps')

        predictions = model.predict_generator(
            generator_strip_xy(data_gen_test, true_values),
            steps=steps_test
        )
        # predictions = dl.scalar.inverse_transform(predictions)
        # true_values = dl.scalar.inverse_transform(np.array(true_values).reshape(-1, 1))
        # Save our predictions
        with h5py.File(configs['model']['filename_predictions'], 'w') as hf:
            dset_p = hf.create_dataset('predictions', data=predictions)
            dset_y = hf.create_dataset('true_values', data=true_values)

        plot.plot_results(predictions[:800], true_values[:800])

    def fit_model(self, data_gen_train, steps_per_epoch, configs, val_data):
        """thread worker for model fitting - so it doesn't freeze on jupyter notebook"""
        history = self.model.fit_generator(
            data_gen_train,
            steps_per_epoch=steps_per_epoch,
            epochs=configs['model']['epochs'],
            validation_data=val_data
        )
        plt.plot(history.history['loss'], color='blue', label='train')
        plt.plot(history.history['val_loss'], color='orange', label='validation')
        plt.show()
        self.model.save(configs['model']['filename_model'])
        print('> Model Trained! Weights saved in', configs['model']['filename_model'])
        return

    def load_model(self):
        self.model = lstm.load_network(self.model_path)


def predict_sequences_multiple(model, data, window_size, prediction_len):
    # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[np.newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs


true_values = []


def generator_strip_xy(data_gen_test, true_values):
    for x, y in data_gen_test:
        true_values += list(y)
        yield x


def fit_model_threaded(model, data_gen_train, steps_per_epoch, configs, val_data):
    """thread worker for model fitting - so it doesn't freeze on jupyter notebook"""
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                              write_graph=True, write_images=True)
    model.fit_generator(
        data_gen_train,
        steps_per_epoch=steps_per_epoch,
        epochs=configs['model']['epochs'],
        validation_data=val_data,
        callbacks=[tensorboard]
    )
    model.save(configs['model']['filename_model'])
    print('> Model Trained! Weights saved in', configs['model']['filename_model'])
    return


dl = etl.ETL(y_method="Integer")

dl.create_clean_datafile(
    filename_in=configs['data']['filename'],
    filename_out=configs['data']['filename_clean'],
    batch_size=configs['data']['batch_size'],
    x_window_size=configs['data']['x_window_size'],
    y_window_size=configs['data']['y_window_size'],
    y_col=configs['data']['y_predict_column'],    # TODO change y_col to y_lag, and y_col is useless now.
    filter_cols=configs['data']['filter_columns'],
    normalise=False
)
'''
print('> Generating clean data from:', configs['data']['filename_clean'], 'with batch_size:', configs['data']['batch_size'])

with h5py.File(configs['data']['filename_clean'], 'r') as hf:
    nrows = hf['x'].shape[0]
    ncols = hf['x'].shape[2]
    
ntrain = int(configs['data']['train_test_split'] * nrows)
steps_per_epoch = int(ntrain / configs['data']['batch_size'])
trsize = steps_per_epoch * configs['data']['batch_size']
nval = nrows - ntrain

with h5py.File(configs['data']['filename_clean'], 'r') as hf:
    val_data_x = hf['x'][ntrain:]
    val_data_y = hf['y'][ntrain:]
    if dl.method == 'Integer' or dl.method == "OneHot":
        val_data_y = keras.utils.to_categorical(val_data_y - 1, num_classes=3)

val_data = (val_data_x, val_data_y)
data_gen_train = dl.generate_clean_data(
    configs['data']['filename_clean'],
    size=trsize,
    batch_size=configs['data']['batch_size']
)

print('> Clean data has', nrows, 'data rows. Training on', ntrain, 'rows with', steps_per_epoch, 'steps-per-epoch')

model = lstm.build_cls_network([ncols, 400, 400, 100])
fit_model_threaded(model, data_gen_train, steps_per_epoch, configs, val_data)

ntest = nrows - ntrain
steps_test = int(ntest / configs['data']['batch_size'])
tesize = steps_test * configs['data']['batch_size']

data_gen_test = dl.generate_clean_data(
    configs['data']['filename_clean'],
    size=tesize,
    batch_size=configs['data']['batch_size'],
    start_index=ntrain
)

print('> Testing model on', ntest, 'data rows with', steps_test, 'steps')

predictions = model.predict_generator(
    generator_strip_xy(data_gen_test, true_values),
    steps=steps_test
)
# predictions = dl.scalar.inverse_transform(predictions)
# true_values = dl.scalar.inverse_transform(np.array(true_values).reshape(-1, 1))
# Save our predictions
with h5py.File(configs['model']['filename_predictions'], 'w') as hf:
    dset_p = hf.create_dataset('predictions', data=predictions)
    dset_y = hf.create_dataset('true_values', data=true_values)
    
plot.plot_results(predictions[:800], true_values[:800])
'''
'''
# Reload the data-generator
data_gen_test = dl.generate_clean_data(
    configs['data']['filename_clean'],
    batch_size=800,
    start_index=ntrain
)
data_x, true_values = next(data_gen_test)
window_size = 50  # numer of steps to predict into the future

# We are going to cheat a bit here and just take the next 400 steps from the
# testing generator and predict that data in its whole
predictions_multiple = predict_sequences_multiple(
    model,
    data_x,
    data_x[0].shape[0],
    window_size
)

plot_results_multiple(predictions_multiple, true_values, window_size)
'''
