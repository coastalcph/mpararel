import logging


def get_logger(name, filename=None, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if filename is not None:
        fh = logging.FileHandler(filename)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger
