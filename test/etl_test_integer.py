import unittest
from etl import ETL


# TODO create a unittest for etl
class TestInteger(unittest.TestCase):
    def setUp(self):
        self.dl = ETL("Integer")

    def test_clean_data(self):
        data_gen = self.dl.clean_data(filepath=["D:\\Multi-LSTM\\test\\test_data.h5"],
                                      batch_size=5,
                                      x_window_size=10,
                                      y_window_size=1,
                                      y_lag=1,
                                      filter_cols=["INNER_CODE", "DATE", "alpha001", "alpha002", "fwd_rtn"])
        x, y = data_gen.__next__()
        self.assertEqual(x.shape, (5, 10, 5))


if __name__ == '__main__':
    unittest.main(verbosity=2)

