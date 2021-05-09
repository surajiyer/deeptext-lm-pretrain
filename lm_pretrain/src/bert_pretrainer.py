from lm_pretrain.src.datasets.chats import ChatDataset
from lm_pretrain.src.my_tensorboard_callback import MyTensorBoardCallback
from lm_pretrain.src.param_parser import DataArguments, ModelArguments, TrainingArguments
import math
import os
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoTokenizer,
    CONFIG_MAPPING,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    set_seed,
    Trainer,
)
from transformers.integrations import TensorBoardCallback
from typing import List, Tuple
import gc


class BERTPreTrainer(object):

    def __init__(
        self,
        data_args: DataArguments,
        model_args: ModelArguments,
        training_args: TrainingArguments,
    ):
        set_seed(training_args.seed)
        self.data_args = data_args
        self.model_args = model_args
        self.training_args = training_args
        self._setup()

    def _setup_tokenizer(self):
        if self.model_args.tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_args.tokenizer_name,
                cache_dir=self.model_args.cache_dir)
        elif self.model_args.model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_args.model_name_or_path,
                cache_dir=self.model_args.cache_dir)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
                "and load it from here, using --tokenizer_name"
            )

    def _setup_model(self):
        """
        Load pretrained model
        """
        if self.model_args.config_name:
            config = AutoConfig.from_pretrained(
                self.model_args.config_name,
                cache_dir=self.model_args.cache_dir)
        elif self.model_args.model_name_or_path:
            config = AutoConfig.from_pretrained(
                self.model_args.model_name_or_path,
                cache_dir=self.model_args.cache_dir)
        else:
            config = CONFIG_MAPPING[self.model_args.model_type]()
            logging.warning("You are instantiating a new config instance from scratch.")

        if self.model_args.model_name_or_path:
            model = AutoModelForPreTraining.from_pretrained(
                self.model_args.model_name_or_path,
                from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
                config=config,
                cache_dir=self.model_args.cache_dir,
            )
        else:
            logging.info("Training new model from scratch")
            model = AutoModelForPreTraining.from_config(config)

        model.resize_token_embeddings(len(self.tokenizer))

        if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not self.data_args.mlm:
            raise ValueError(
                "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
                "flag (masked language modeling)."
            )

        self.model = model

    def _setup_trainer(self):
        def get_dataset(args: DataArguments, tokenizer: PreTrainedTokenizer, evaluate: bool = False):
            file_path = args.eval_data_file if evaluate else args.train_data_file
            print("Performing line by line tokenization")
            return ChatDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)

        def get_data_collator(examples: List[Tuple[torch.Tensor, int]]):
            inputs, is_next_random = zip(*examples)
            output = mlm_collator(inputs)
            output.update({
                'next_sentence_label': torch.tensor(is_next_random, dtype=torch.long)
            })
            return output

        # Get datasets
        train_dataset = (
            get_dataset(self.data_args, tokenizer=self.tokenizer)
            if self.training_args.do_train else None
        )
        eval_dataset = (
            get_dataset(self.data_args, tokenizer=self.tokenizer, evaluate=True)
            if self.training_args.do_eval else None
        )
        mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=self.data_args.mlm,
            mlm_probability=self.data_args.mlm_probability)

        # Initialize our Trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            data_collator=get_data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[MyTensorBoardCallback,],
        )

        # todo: why there's a another folder in log_subdir
        # todo: how to replace huggingface Trainer abstraction with explicit training loops?
        # todo: add more metrics to tensorboard
        self.trainer.remove_callback(TensorBoardCallback)

    def _setup(self):
        self._setup_tokenizer()
        self._setup_model()
        self._setup_trainer()

    def fit(self):
        # create experiment output directory
        # under output_dir
        if os.path.split(self.training_args.output_dir)[1] != f"experiment_{self.training_args.now}":
            self.training_args.output_dir = os.path.join(
                self.training_args.output_dir,
                f"experiment_{self.training_args.now}")
        os.makedirs(self.training_args.output_dir)

        if (
            os.path.exists(self.training_args.output_dir)
            and os.listdir(self.training_args.output_dir)
            and self.training_args.do_train
            and not self.training_args.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({self.training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )

        if self.data_args.block_size <= 0:
            self.data_args.block_size = self.tokenizer.model_max_length
        else:
            self.data_args.block_size = min(self.data_args.block_size, self.tokenizer.model_max_length)

        # Training
        model_path = (
            self.model_args.model_name_or_path
            if self.model_args.model_name_or_path is not None and os.path.isdir(self.model_args.model_name_or_path)
            else None
        )
        gc.collect()
        torch.cuda.empty_cache()
        self.trainer.train(model_path=model_path)
        self.trainer.save_model()

        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if self.trainer.is_world_master():
            self.tokenizer.save_pretrained(self.training_args.output_dir)

    def evaluate(self):
        # create experiment output directory
        # under output_dir
        if os.path.split(self.training_args.output_dir)[1] != f"experiment_{self.training_args.now}":
            self.training_args.output_dir = os.path.join(
                self.training_args.output_dir,
                f"experiment_{self.training_args.now}")
        os.makedirs(self.training_args.output_dir)

        if self.data_args.eval_data_file is None and self.training_args.do_eval:
            raise ValueError(
                "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
                "or remove the --do_eval argument."
            )

        results = {}
        logging.info("*** Evaluate ***")

        eval_output = self.trainer.evaluate()

        # Perplexity here is e^(eval_losses/n)
        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(self.training_args.output_dir, "eval_results_lm.txt")
        if self.trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logging.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logging.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

        return results
