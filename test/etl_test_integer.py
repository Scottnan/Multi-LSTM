import unittest
import h5py
import os
import warnings
from etl import ETL
warnings.filterwarnings("ignore")


class TestInteger(unittest.TestCase):
    def setUp(self):
        self.dl = ETL("Integer")

    def test_clean_data(self):
        data_gen = self.dl.clean_data(filepath=["test_data_part1.h5",
                                                "test_data_part2.h5",
                                                "test_data_part3.h5",
                                                "test_data_part4.h5",
                                                "test_data_part5.h5"],
                                      batch_size=5,
                                      x_window_size=10,
                                      y_window_size=1,
                                      y_lag=1,
                                      filter_cols=["INNER_CODE", "DATE", "alpha001", "alpha002", "fwd_rtn"])
        x, y = data_gen.__next__()
        self.assertEqual(x.shape, (5, 10, 5))
        self.assertEqual(y.shape, (5,))
        self.assertEquals(x[1, 0, 0], 3)
        self.assertEquals(x[1, 0, 1], 16442.0)
        self.assertAlmostEquals(x[1, 0, 2], -1.26353, 5)
        self.assertAlmostEquals(x[1, 0, 3], -0.90511, 5)
        self.assertEqual(x[0, 0, 4], 1.0)
        self.assertEqual(x[1, 0, 4], 2.0)
        self.assertEqual(y[0], 2.0)
        self.assertEqual(y[1], 0.0)

    def test_create_clean_datafile(self):
        self.dl.create_clean_datafile(filename_in=["test_data_part1.h5",
                                                   "test_data_part2.h5",
                                                   "test_data_part3.h5",
                                                   "test_data_part4.h5",
                                                   "test_data_part5.h5"],
                                      filename_out="tmp_clean_data.h5",
                                      batch_size=5,
                                      x_window_size=10)
        clean_data = h5py.File("tmp_clean_data.h5", "r")
        _x = clean_data['x'][:]
        _y = clean_data['y'][:]
        self.assertEqual(_x.shape, (195, 10, 5))
        self.assertEqual(_y.shape, (195,))
        self.assertListEqual(_y[:5].tolist(), [2.0, 0.0, 2.0, 1.0, 1.0])

    def test_generate_clean_data(self):
        data_gen1 = self.dl.generate_clean_data(filename="tmp_clean_data.h5",
                                                size=15,
                                                batch_size=5)
        _x, _y = data_gen1.__next__()
        self.assertEqual(_x.shape, (5, 10, 3))
        self.assertEqual(_y.shape, (5, 3))
        self.assertAlmostEquals(_x[1, 0, 0], -1.26353, 5)
        self.assertAlmostEquals(_x[1, 0, 1], -0.90511, 5)
        self.assertEqual(_x[1, 0, 2], 2.0)
        self.assertEqual(_x[0, 0, 2], 1.0)
        self.assertListEqual(_y[0].tolist(), [0, 0, 1])
        self.assertListEqual(_y[1].tolist(), [1, 0, 0])
        self.assertListEqual(_y[2].tolist(), [0, 0, 1])
        self.assertListEqual(_y[3].tolist(), [0, 1, 0])
        self.assertListEqual(_y[4].tolist(), [0, 1, 0])
        _x, _y = data_gen1.__next__()
        self.assertListEqual(_y[0].tolist(), [0, 1, 0])
        self.assertListEqual(_y[1].tolist(), [1, 0, 0])
        self.assertListEqual(_y[2].tolist(), [0, 0, 1])
        self.assertListEqual(_y[3].tolist(), [0, 1, 0])
        self.assertListEqual(_y[4].tolist(), [0, 0, 1])
        _x, _y = data_gen1.__next__()
        self.assertListEqual(_y[0].tolist(), [1, 0, 0])
        self.assertListEqual(_y[1].tolist(), [0, 0, 1])
        self.assertListEqual(_y[2].tolist(), [1, 0, 0])
        self.assertListEqual(_y[3].tolist(), [0, 1, 0])
        self.assertListEqual(_y[4].tolist(), [1, 0, 0])
        _x, _y = data_gen1.__next__()
        self.assertListEqual(_y[0].tolist(), [0, 0, 1])
        self.assertListEqual(_y[1].tolist(), [1, 0, 0])
        self.assertListEqual(_y[2].tolist(), [0, 0, 1])
        self.assertListEqual(_y[3].tolist(), [0, 1, 0])
        self.assertListEqual(_y[4].tolist(), [0, 1, 0])
        data_gen2 = self.dl.generate_clean_data(filename="tmp_clean_data.h5",
                                                size=10,
                                                batch_size=5,
                                                start_index=5)
        _x, _y = data_gen2.__next__()
        self.assertListEqual(_y[0].tolist(), [0, 1, 0])
        self.assertListEqual(_y[1].tolist(), [1, 0, 0])
        self.assertListEqual(_y[2].tolist(), [0, 0, 1])
        self.assertListEqual(_y[3].tolist(), [0, 1, 0])
        self.assertListEqual(_y[4].tolist(), [0, 0, 1])
        _x, _y = data_gen2.__next__()
        self.assertListEqual(_y[0].tolist(), [1, 0, 0])
        self.assertListEqual(_y[1].tolist(), [0, 0, 1])
        self.assertListEqual(_y[2].tolist(), [1, 0, 0])
        self.assertListEqual(_y[3].tolist(), [0, 1, 0])
        self.assertListEqual(_y[4].tolist(), [1, 0, 0])
        _x, _y = data_gen2.__next__()
        self.assertListEqual(_y[0].tolist(), [0, 1, 0])
        self.assertListEqual(_y[1].tolist(), [1, 0, 0])
        self.assertListEqual(_y[2].tolist(), [0, 0, 1])
        self.assertListEqual(_y[3].tolist(), [0, 1, 0])
        self.assertListEqual(_y[4].tolist(), [0, 0, 1])

    @classmethod
    def tearDownClass(cls):
        os.remove("tmp_clean_data.h5")


if __name__ == '__main__':
    unittest.main()
