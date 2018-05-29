import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib


class ETL(object):
    """Extract Transform Load class for all data operations pre model inputs. Data is read in generative way to allow
    for large datafiles and low memory utilisation"""
    def generate_clean_data(self, filename, size, batch_size=1000, start_index=0):
        # TODO check the size/batch_size is int?
        self.scalar = joblib.load('model/scalar.pkl')
        with h5py.File(filename, 'r') as hf:
            i = start_index
            while True:
                if i % size < (i + batch_size) % size:
                    data_x = hf['x'][i % size: (i + batch_size) % size]
                    data_y = hf['y'][i % size: (i + batch_size) % size]
                else:
                    data_x = hf['x'][i % size: size]
                    data_y = hf['y'][i % size: size]
                i += batch_size
                yield (data_x, data_y)

    def create_clean_datafile(self, filename_in, filename_out, batch_size=1000, x_window_size=100, y_window_size=1,
                              y_col=0, filter_cols=None, normalise=False):
        """Incrementally save a datafile of clean data ready for loading straight into model"""
        print('> Creating x & y data files...')

        data_gen = self.clean_data(
            filename_in,
            batch_size=batch_size,
            x_window_size=x_window_size,
            y_window_size=y_window_size,
            y_col=y_col,
            filter_cols=filter_cols,
            normalise=normalise
        )

        i = 0
        with h5py.File(filename_out, 'w') as hf:
            x1, y1 = next(data_gen)
            # Initialise hdf5 x, y datasets with first chunk of data
            rcount_x = x1.shape[0]
            dset_x = hf.create_dataset('x', shape=x1.shape, maxshape=(None, x1.shape[1], x1.shape[2]), chunks=True)
            dset_x[:] = x1
            rcount_y = y1.shape[0]
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

    def clean_data(self, filepath, batch_size, x_window_size, y_window_size, y_col, filter_cols, normalise):
        """Cleans and Normalises the data in batches `batch_size` at a time"""
        raw_data = pd.read_hdf(filepath)
        # codes = set(raw_data['INNER_CODE'].tolist())

        if filter_cols:
            # Remove any columns from data that we don't need by getting the difference between cols and filter list
            rm_cols = set(raw_data.columns) - set(filter_cols)
            for col in rm_cols:
                del raw_data[col]

        # Convert y-predict column name to numerical index

        x_data = []
        y_data = []
        j = 0  # The number of sample
        raw_data.dropna(inplace=True)
        scalar = MinMaxScaler(feature_range=(0, 1))
        self.scalar = scalar.fit(raw_data['fwd_rtn'].reshape(-1, 1))   # TODO future function
        joblib.dump(self.scalar, 'model/scalar.pkl')
        raw_data['fwd_rtn'] = self.scalar.transform(raw_data['fwd_rtn'].reshape(-1, 1))
        # Each stock feature is scrolled as sample data
        for code in set(raw_data['INNER_CODE']):
            data = raw_data[raw_data['INNER_CODE'] == code]
            data['rtn'] = data['fwd_rtn'].shift()   # Shift 1 period
            data.drop("fwd_rtn", axis=1, inplace=True)
            data.drop("INNER_CODE", axis=1, inplace=True)
            num_rows = len(data)
            y_col = list(data.columns).index('rtn')
            print('> Creating x & y data files | Code:', code)
            i = 0
            while (i + x_window_size + y_window_size) <= num_rows:
                x_window_data = data[i:(i + x_window_size)]
                y_window_data = data[(i + x_window_size):(i + x_window_size + y_window_size)]

                # Remove any windows that contain NaN
                if x_window_data.isnull().values.any() or y_window_data.isnull().values.any():
                    i += 1
                    continue

                if normalise:
                    abs_base, x_window_data = self.zero_base_standardise(x_window_data)
                    _, y_window_data = self.zero_base_standardise(y_window_data, abs_base=abs_base)

                # Average of the desired predicter y column
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

    def zero_base_standardise(self, data, abs_base=pd.DataFrame()):
        """Standardise dataframe to be zero based percentage returns from i=0"""
        if abs_base.empty:
            abs_base = data.iloc[0]
        data_standardised = (data / abs_base) - 1
        return abs_base, data_standardised

    def min_max_normalise(self, data, data_min=pd.DataFrame(), data_max=pd.DataFrame()):
        """Normalise a Pandas dataframe using column-wise min-max normalisation (can use custom min, max if desired)"""
        if data_min.empty:
            data_min = data.min()
        if data_max.empty:
            data_max = data.max()
        data_normalised = (data - data_min) / (data_max - data_min)
        return data_min, data_max, data_normalised
