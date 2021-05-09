import logging
from texttable import Texttable


def setup_logging(*, filepath=None, level=logging.INFO):
    """Setup logging"""
    config = {
        'level': level,
        'format': '%(asctime)s %(filename)s:%(lineno)d %(levelname)s  %(message)s',
        'datefmt': '%Y-%m-%d %X',
        'handlers': (
            [logging.StreamHandler(),]
            + ([logging.FileHandler(filepath),] if filepath else [])
        )
    }
    logging.basicConfig(**config)


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows(
        [["Parameter", "Value"]]
        + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    logging.info(t.draw())
