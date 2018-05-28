import logging
import os


def log_config(filename, output_level):
    path = os.path.join(os.getcwd(), 'log')
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    logging.basicConfig(
        filename=os.path.join(path, filename),
        level=exec("logging.{}".format(output_level)),
        format='%(asctime)s : %(levelname)s  %(message)s',
        datefmt='%m-%d %H:%M:%S',
    )
