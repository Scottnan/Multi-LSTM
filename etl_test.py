import unittest
from etl import ETL


class TestETL(unittest.TestCase):
    def test_init(self):
        dl = ETL()

