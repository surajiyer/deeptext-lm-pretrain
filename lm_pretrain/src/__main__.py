from lm_pretrain.src.bert_pretrainer import BERTPreTrainer
from lm_pretrain.src.exBert import exBERTPreTrainer
from lm_pretrain.src.param_parser import parameter_parser
from lm_pretrain.src.utils import setup_logging, tab_printer
import logging
import os


def main():
    # parse and print args
    args = parameter_parser()

    # setup logging
    log_level = logging.INFO if args[2].local_rank in [-1, 0] else logging.WARN
    setup_logging(
        filepath=os.path.join(args[2].logging_dir, f"{args[2].now}.log"),
        level=log_level)

    # log input arguments
    logging.info("Data Args:")
    tab_printer(args[0])
    logging.info("Model Args:")
    tab_printer(args[1])
    logging.info("Training Args:")
    tab_printer(args[2])

    # pretrain BERT
    if args[1].method == "bert":
        model = BERTPreTrainer(*args)
    elif args[1].method == "exbert":
        model = exBERTPreTrainer(*args)
    else:
        raise ValueError(f"Unknown method given: {args[1].method}")
    model.fit()
    model.evaluate()


if __name__ == '__main__':
    main()
