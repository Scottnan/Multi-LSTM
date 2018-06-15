import os
import time
import gru
import etl
import json
import plot_utils as plot
import keras
import datetime
import h5py
from keras.callbacks import TensorBoard, EarlyStopping
configs = json.loads(open('configs.json').read())
tstart = time.time()
log_time = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')


class MulLSTM(object):
    def __init__(self, configs):
        self.method = configs['model']['method']
        self.model_path = configs['model']['filename_model']
        self.raw_path = configs['data']['filename']
        self.clean_path = configs['data']['filename_clean']
        self.batch_size = configs['data']['batch_size']
        self.x_window_size = configs['data']['x_window_size']
        self.y_window_size = configs['data']['y_window_size']
        self.y_lag = configs['data']['y_lag']
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
                y_lag=self.y_lag,
                filter_cols=self.filter_cols
            )
            print('> Generating clean data from:', self.clean_path, 'with batch_size:', self.batch_size)

    def fit(self):
        with h5py.File(self.clean_path, 'r') as hf:
            self.fit_nrows = hf['x'].shape[0]
            self.fit_ncols = hf['x'].shape[2]

        self.ntrain = int(self.train_test_split * self.fit_nrows)
        steps_per_epoch = int(self.ntrain / self.batch_size)
        trsize = steps_per_epoch * self.batch_size

        with h5py.File(self.clean_path, 'r') as hf:
            val_data_x = hf['x'][self.ntrain:, :, 2:]
            val_data_y = hf['y'][self.ntrain:]
            if self.dl.method == 'Integer' or self.dl.method == "OneHot":
                val_data_y = keras.utils.to_categorical(val_data_y, num_classes=2)

        val_data = (val_data_x, val_data_y)
        data_gen_train = self.dl.generate_clean_data(
            configs['data']['filename_clean'],
            size=trsize,
            batch_size=configs['data']['batch_size']
        )

        print('> Clean data has', self.fit_nrows, 'data rows. Training on', self.ntrain, 'rows with', steps_per_epoch,
              'steps-per-epoch')

        self.model = gru.build_cls_network([self.fit_ncols - 2, 128, 64, 64, 32])
        self.model.summary()
        self.fit_model(data_gen_train, steps_per_epoch, configs, val_data)

    def validation(self):
        ntest = self.fit_nrows - self.ntrain
        steps_test = int(ntest / configs['data']['batch_size'])
        tesize = steps_test * configs['data']['batch_size']

        data_gen_test = self.dl.generate_clean_data(
            configs['data']['filename_clean'],
            size=tesize,
            batch_size=configs['data']['batch_size'],
            start_index=self.ntrain
        )

        print('> Testing model on', ntest, 'data rows with', steps_test, 'steps')

        predictions = self.model.predict_generator(
            self.generator_strip_xy(data_gen_test),
            steps=steps_test
        )
        # Save our predictions
        with h5py.File(configs['model']['filename_predictions'], 'w') as hf:
            dset_p = hf.create_dataset('predictions', data=predictions)
            dset_y = hf.create_dataset('true_values', data=self.ture_values)

        plot.plot_results(predictions[:800], self.ture_values[:800])

    def fit_model(self, data_gen_train, steps_per_epoch, configs, val_data):
        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                                  write_graph=True, write_images=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        self.model.fit_generator(
            data_gen_train,
            steps_per_epoch=steps_per_epoch,
            epochs=configs['model']['epochs'],
            validation_data=val_data,
            callbacks=[tensorboard, early_stopping]
        )
        self.model.save(configs['model']['filename_model'])
        print('> Model Trained! Weights saved in', configs['model']['filename_model'])
        return

    def load_model(self):
        self.model = gru.load_network(self.model_path)

    def generator_strip_xy(self, data_gen_test):
        self.ture_values = []
        for x, y in data_gen_test:
            self.ture_values += list(y)
            yield x

    def generator_train_data_for_test(self):
        if os.path.exists(self.clean_path):
            print('> Clean data exists!')
        else:
            self.dl.create_clean_datafile(
                filename_in=self.raw_path,
                filename_out=self.clean_path,
                batch_size=self.batch_size,
                x_window_size=self.x_window_size,
                y_window_size=self.y_window_size,
                y_lag=self.y_lag,
                filter_cols=self.filter_cols,
                for_test=True
            )
            print('> Generating clean data from:', self.clean_path, 'with batch_size:', self.batch_size)


if __name__ == "__main__":
    model = MulLSTM(configs)
    model.clean_data()
    # model.generator_train_data_for_test()
    model.fit()
