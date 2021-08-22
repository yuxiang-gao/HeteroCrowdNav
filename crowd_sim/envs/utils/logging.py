import logging
import sys


def logging_init(level=logging.INFO, output=None, resume=None) -> None:
    # formatter = logging.Formatter(
    #     "%(asctime)s\t%(levelname)s -- %(filename)s:%(lineno)s -- %(message)s",
    #     "%m-%d %H:%M:%S",
    # )

    format_str = "%(asctime)s %(levelname)s[%(module)s:%(lineno)s]: %(message)s"
    data_fmt = "%m-%d %H:%M:%S"

    mode = "a" if resume else "w"

    if output is not None:
        console_handler = logging.StreamHandler(sys.stdout)
        file_handler = logging.FileHandler(output, mode=mode)

        logging.basicConfig(
            level=level,
            handlers=[console_handler, file_handler],
            format=format_str,
            datefmt=data_fmt,
        )
    else:
        logging.basicConfig(
            level=level,
            format=format_str,
            datefmt=data_fmt,
        )
