import logging

from tqdm.contrib.logging import logging_redirect_tqdm


def logging_debug(*msg):
    with logging_redirect_tqdm():
        logging.debug(*msg)


def logging_info(*msg):
    with logging_redirect_tqdm():
        logging.info(*msg)
