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


def debug(msg):
    exec("logging.debug('{}')".format(msg))


def info(msg):
    exec("logging.info('{}')".format(msg))


def warning(msg):
    exec("logging.warning('{}')".format(msg))


def error(msg):
    exec("logging.error('{}')".format(msg))
