import argparse
from argparse import RawTextHelpFormatter
from dataclasses import dataclass, field
from datetime import datetime
import re
from transformers import (
    MODEL_FOR_PRETRAINING_MAPPING,
    TrainingArguments
)
from typing import Optional
import yaml


MODEL_CONFIG_CLASSES = list(MODEL_FOR_PRETRAINING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


@dataclass
class exBERTModelArguments(ModelArguments):
    """
    exBERT model specific arguments.
    """

    train_extension_only: bool = field(
        default=True, metadata={"help": "Whether to train only the extended vocabulary."}
    )
    config_ADD_name: Optional[str] = field(
        default=None, metadata={"help": "Additional configuration name for exBERT"}
    )


def parameter_parser():
    """
    A method to parse YAML parameters.
    """
    # get path to YAML file as argument
    parser = argparse.ArgumentParser(description="Run BERT pretraining.", formatter_class=RawTextHelpFormatter)
    parser.add_argument("-s",
                        "--settings",
                        default="settings.yml",
                        help="Path to settings.yml file")
    parser.add_argument("--now",
                        type=str,
                        default=datetime.now().strftime("%Y%m%d_%H%M"),
                        help="Current timestamp")
    parser.add_argument("--method",
                        type=str,
                        default="bert",
                        help="Method of pre-training:\n(1) 'bert': Regular BERT pretraining.\n(2) 'exbert': exBERT pretraining")
    args = parser.parse_args()
    yaml_path = args.settings

    # yaml loader float fix: https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    # Load settings
    with open(yaml_path, "r") as stream:
        settings = yaml.load(stream, Loader=loader)

    training_args = TrainingArguments(**settings["training_args"])
    training_args.now = args.now
    data_args = DataArguments(**settings["data_args"])
    if args.method == "bert":
        model_args = ModelArguments(**settings["model_args"])
    elif args.method == "exbert":
         model_args = exBERTModelArguments(**settings["model_args"])
    else:
        raise ValueError(f"Unknown method given: {args.method}")
    model_args.method = args.method

    return data_args, model_args, training_args
