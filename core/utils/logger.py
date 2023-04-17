import logging


class Logger(object):
    """
    Helper class for logging.
    Arguments:
        path (str): Path to log file.
    """
    def __init__(self, path):
        self.path = path
        self.f = open(path,"w")
        print('Logging to file: ', self.path)

    def log(self, message):
        print(message,file=self.f,flush=True)
        print(message)
