import h5py
import numpy as np
import pandas as pd
import keras
from math import isnan
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib


class ETL(object):
    def __init__(self, y_method="Numeric"):
        assert y_method in ["Numeric", "MinMax", "Integer", "OneHot"], "y method must in \"Numeric\", \"MinMax\", " \
                                                                       "\"Integer\", \"OneHot\""
        self.method = y_method
    """Extract Transform Load class for all data operations pre model inputs. Data is read in generative way to allow
    for large datafiles and low memory utilisation"""
    def generate_clean_data(self, filename, size, batch_size=1000, start_index=0):
        assert (size / batch_size).is_integer(), "Size / Batch size is not an integer!"
        if self.method == "MinMax":
            self.scalar = joblib.load('model/scalar.pkl')
        with h5py.File(filename, 'r') as hf:
            i = 0
            while True:
                if i % size < (i + batch_size) % size:
                    data_x = hf['x'][start_index + i % size: start_index + (i + batch_size) % size, :, 2:]
                    data_y = hf['y'][start_index + i % size: start_index + (i + batch_size) % size]
                else:
                    data_x = hf['x'][start_index + i % size: start_index + size, :, 2:]
                    data_y = hf['y'][start_index + i % size: start_index + size]
                if self.method == 'Integer' or self.method == "OneHot":
                    data_y = keras.utils.to_categorical(data_y, num_classes=3)
                i += batch_size
                yield (data_x, data_y)

    def create_clean_datafile(self, filename_in, filename_out, batch_size=1000, x_window_size=100, y_window_size=1,
                              y_lag=1, filter_cols=None):
        """Incrementally save a datafile of clean data ready for loading straight into model"""
        print('> Creating x & y data files...')

        data_gen = self.clean_data(
            filepath=filename_in,
            batch_size=batch_size,
            x_window_size=x_window_size,
            y_window_size=y_window_size,
            y_lag=y_lag,
            filter_cols=filter_cols
        )

        i = 0
        with h5py.File(filename_out, 'w') as hf:
            x1, y1 = next(data_gen)
            # Initialise hdf5 x, y datasets with first chunk of data
            rcount_x = x1.shape[0]
            dset_x = hf.create_dataset('x', shape=x1.shape, maxshape=(None, x1.shape[1], x1.shape[2]), chunks=True)
            dset_x[:] = x1
            rcount_y = y1.shape[0]
            if self.method == "OneHot":
                dset_y = hf.create_dataset('y', shape=y1.shape, maxshape=(None, y1.shape[1], y1.shape[2]), chunks=True)
            else:
                dset_y = hf.create_dataset('y', shape=y1.shape, maxshape=(None,), chunks=True)
            dset_y[:] = y1

            for x_batch, y_batch in data_gen:
                # Append batches to x, y hdf5 datasets
                print('> Creating x & y data files | Batch:', i, end='\r')
                dset_x.resize(rcount_x + x_batch.shape[0], axis=0)
                dset_x[rcount_x:] = x_batch
                rcount_x += x_batch.shape[0]
                dset_y.resize(rcount_y + y_batch.shape[0], axis=0)
                dset_y[rcount_y:] = y_batch
                rcount_y += y_batch.shape[0]
                i += 1

        print('> Clean datasets created in file `' + filename_out + '.h5`')

    def clean_data(self, filepath, batch_size, x_window_size, y_window_size, y_lag, filter_cols):
        """Cleans the data in batches `batch_size` at a time"""
        f = []
        for file in filepath:
            f.append(pd.read_hdf(file))
        raw_data = pd.concat(f)

        if filter_cols:
            # Remove any columns from data that we don't need by getting the difference between cols and filter list
            rm_cols = set(raw_data.columns) - set(filter_cols)
            for col in rm_cols:
                del raw_data[col]

        # Convert y-predict column name to numerical index

        x_data = []
        y_data = []

        j = 0  # The number of sample
        if self.method == "MinMax":
            raw_data.dropna(inplace=True)
            scalar = MinMaxScaler(feature_range=(0, 1))
            self.scalar = scalar.fit(raw_data['fwd_rtn'].reshape(-1, 1))   # TODO future function
            joblib.dump(self.scalar, 'model/scalar.pkl')
            raw_data['fwd_rtn'] = self.scalar.transform(raw_data['fwd_rtn'].reshape(-1, 1))
            raw_data.drop("DATE", axis=1, inplace=True)

        if self.method == "Integer" or self.method == "OneHot":
            tmp = pd.DataFrame()
            date = set(raw_data.DATE)
            for d in date:
                data = raw_data[raw_data.DATE == d]
                if self.method == "OneHot":
                    self.y2integer(data)
                else:
                    self.y2integer(data)  # useless if
                tmp = pd.concat([tmp, data])
            raw_data = tmp
            raw_data.sort_values(by="DATE", inplace=True)

        # Each stock feature is scrolled as sample data
        for code in set(raw_data['INNER_CODE']):
            data = raw_data[raw_data['INNER_CODE'] == code]
            data['rtn'] = data['fwd_rtn'].shift()   # Shift 1 period
            data.drop("fwd_rtn", axis=1, inplace=True)
            y_col = list(data.columns).index('rtn')
            """
                        if self.method == "OneHot":
                data.dropna(inplace=True)
                one_hot = keras.utils.to_categorical(data['rtn'] - 1, num_classes=5)
                data.drop("rtn", axis=1, inplace=True)
                old_col = data.columns
                data = pd.concat([data, pd.DataFrame(one_hot)], axis=1)
                new_col = data.columns
                one_col = list(set(new_col) - set(old_col))
                # y_col = [list(data.columns).index(c) for c in one_col].sort()   # TODO FIX BUG one_hot columns
            else:
                y_col = list(data.columns).index('rtn')
            """
            num_rows = len(data)
            print('> Creating x & y data files | Code:', code)
            i = 0
            while (i + x_window_size + y_window_size) <= num_rows:
                x_window_data = data[i:(i + x_window_size)]
                y_window_data = data[(i + x_window_size + y_lag - 1):(i + x_window_size + y_window_size + y_lag - 1)]

                # Remove any windows that contain NaN
                if x_window_data.isnull().values.any() or y_window_data.isnull().values.any():
                    i += 1
                    continue

                if self.method == "OneHot":
                    y_average = y_window_data.values[:, -5:]
                else:
                    y_average = np.average(y_window_data.values[:, y_col])
                x_data.append(x_window_data.values)
                y_data.append(y_average)
                i += 1
                j += 1

                # Restrict yielding until we have enough in our batch. Then clear x, y data for next batch
                if j % batch_size == 0:
                    # Convert from list to 3 dimensional numpy array [windows, window_val, val_dimension]
                    x_np_arr = np.array(x_data)
                    y_np_arr = np.array(y_data)
                    x_data = []
                    y_data = []
                    yield (x_np_arr, y_np_arr)

    @staticmethod
    def y2_2class(data, per=0.3):
        """
        :param data: fwd_twn series
        :param per:
        :return: ratio <= per        x=1
                 ratio >= 1-per      x=0
                 per < ratio < 1-per x=na
        """
        def fun(x):
            if isnan(x) or (x >= per or x <= 1 - per):
                return np.nan
            elif x <= per:
                return 1
            else:
                return 0
        data['fwd_rtn'] = (data.fwd_rtn.rank(ascending=False) / len(data)).apply(fun)

    @staticmethod
    def y2integer(data, cate=3):
        def fun(x):
            if isnan(x):
                return x
            else:
                return int(x / (1 / cate))
        # it will return a rank ratio not the origin forward return
        data['fwd_rtn'] = ((data.fwd_rtn.rank(ascending=False) - 1) / len(data)).apply(fun)
