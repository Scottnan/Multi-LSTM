import gru
import etl
import h5py
import json
configs = json.loads(open('configs.json').read())
model = gru.load_network("model/model_saved_2016_20180614.h5")
dl = etl.ETL("Integer")
dl.usage = "test"

true_values = []


def generator_strip_xy(data_gen_test, true_values):
    for x, y in data_gen_test:
        true_values += list(y)
        yield x


with h5py.File(configs['data']['filename_clean'], 'r') as hf:
    nrows = hf['x'].shape[0]
    ncols = hf['x'].shape[2]
ntest = nrows
steps_test = int(ntest / configs['data']['batch_size'])
tesize = steps_test * configs['data']['batch_size']
data_gen_test = dl.generate_clean_data(
    configs['data']['filename_clean'],
    size=tesize,
    batch_size=configs['data']['batch_size']
)
print('> Testing model on', ntest, 'data rows with', steps_test, 'steps')
tesize = steps_test * configs['data']['batch_size']
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
