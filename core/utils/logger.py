import logging


class Logger(object):
    """
    Helper class for logging.
    Arguments:
        path (str): Path to log file.
    """
    def __init__(self, path):
        self.path = path
        print('Logging to file: ', self.path)

    def log(self, message):
        print(message)
