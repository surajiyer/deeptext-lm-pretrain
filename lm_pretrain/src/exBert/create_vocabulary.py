import click
from glob import glob
import logging
import os
from sentencepiece import SentencePieceTrainer
# import tempfile
from transformers import BertTokenizerFast
import uuid


def extract_sentencepiece_vocab(
    src_txt_glob_path,
    model_prefix='sentencepiece',
    num_threads=24,
    vocab_size=10_001,
):
    """
    Extract vocabulary using SentencePiece with Unigram model.

    Arguments
        src_txt_glob_path: str
            Input files pathname pattern, e.g., /path/to/files/*_abc_*.txt.

        model_prefix: str, default="sentencepiece"
            Output model name prefix. <model_prefix>.model and <model_prefix>.vocab are generated.

        num_threads: int, default=24
            Number of threads to use.

        vocab_size: int, default=10001
            Output vocabulary size.
    """
    input_paths = ','.join(list(glob(src_txt_glob_path)))
    logging.info('Input Path(s):\n', input_paths.replace(',', '\n'), sep='')

    cmd = '--input={} --vocab_size={} --num_threads={}'\
    ' --shuffle_input_sentence=true --model_type=unigram --split_by_number=false'\
    ' --split_by_unicode_script=false --model_prefix={} --bos_piece=[CLS]'\
    ' --eos_piece=[SEP] --unk_piece=[UNK] --control_symbols=[PAD],[MASK]'.format(
        input_paths, vocab_size, num_threads, model_prefix
    )
    logging.info(cmd)
    trainer = SentencePieceTrainer.Train(cmd)


def convert_vocab_to_wordpiece(
    source_path="sentencepiece.vocab",
    dest_path="wordpiece.vocab",
):
    """
    Convert SentencePiece vocabulary to WordPiece.

    Arguments
        source_path: str, default="sentencepiece.vocab"
            Path to SentencePiece <model_prefix>.vocab.

        dest_path: str, default="wordpiece.vocab"
            Path to WordPiece .vocab file.
    """
    vocab = []

    logging.info('Reading SentencePiece vocabulary from {}'.format(source_path))
    with open(source_path, 'r') as f:
        for l in f.readlines():
            t, _ = l.split('\t')
            if t[0] == '▁':
                t = t[1:]
            elif t[0] != '[' or t[-1] != ']':
                t = '##' + t
            if len(t) > 0:
                vocab.append(t)

    vocab_special = vocab[:5]
    vocab_single_chars = sorted([t for t in vocab[5:] if len(t) == 1]) + sorted([t for t in vocab[5:] if len(t) == 3 and t[:2] == '##'])
    vocab_prefix = sorted([t for t in vocab[5:] if t[:2] != '##' and len(t) > 1])
    vocab_suffix = sorted([t for t in vocab[5:] if t[:2] == '##' and len(t) > 3])

    vocab = vocab_special + vocab_single_chars + vocab_prefix + vocab_suffix

    with open(dest_path, 'w') as f:
        f.write('\n'.join(vocab))

    logging.info('Saved WordPiece vocabulary to {}'.format(dest_path))
    logging.info(f'special={len(vocab_special)}, single_chars={len(vocab_single_chars)}, prefixes={len(vocab_prefix)}, suffixes={len(vocab_suffix)}, total={len(vocab)}')


def merge_original_and_extra_vocab(
    tokenizer_name,
    dest_tokenizer_name=uuid.uuid4().hex[:6].upper(),
    src_wordpiece_vocab="wordpiece.vocab",
    mode="a",
):
    """
    Merge original BERT model vocabulary with our extra vocabulary.

    Arguments
        tokenizer_name: str
            Valid model id on huggingface.co or path to a directory containing vocab.txt file.

        dest_tokenizer_name: str, default=random 6-letter word
            Path to output tokenizer directory with merged vocabulary.

        src_wordpiece_vocab: str, default="wordpiece.vocab"
            Path to WordPiece .vocab file.

        mode: str, default="a"
            mode="a" appends the extra vocabulary to the end of the original.
            mode="w" concatenates and sorts the merged vocabulary.
            In both cases, duplicates will be removed.
    """
    assert mode in ("a", "w")

    # load the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)

    # save a temporary local copy of the tokenizer
    tokenizer.save_pretrained(dest_tokenizer_name)
    original_vocab_path = os.path.join(dest_tokenizer_name, "vocab.txt")

    # load the vocabulary, merge and filter them
    vocab = []

    with open(original_vocab_path, 'r') as f:
        for l in f.readlines():
            vocab.append(l[:-1])

    original_len = len(vocab)

    with open(src_wordpiece_vocab, 'r') as f:
        for i, l in enumerate(f.readlines()):
            if i < 5:
                # skip the first 5 special tokens
                continue
            vocab.append(l[:-1])

    if mode == "w":
        vocab_special = vocab[:5]
        vocab = list(set(vocab[5:]))  # remove duplicates
        vocab_single_chars = sorted([t for t in vocab if len(t) == 1]) + sorted([t for t in vocab if len(t) == 3 and t[:2] == '##'])
        vocab_prefix = sorted([t for t in vocab if t[:2] != '##' and len(t) > 1])
        vocab_suffix = sorted([t for t in vocab if t[:2] == '##' and len(t) > 3])

        vocab = vocab_special + vocab_single_chars + vocab_prefix + vocab_suffix
    elif mode == "a":
        # remove duplicates
        vocab = list(set(vocab))
        vocab_single_chars = (
            sorted([t for t in vocab[original_len:] if len(t) == 1])
            + sorted([t for t in vocab[original_len:] if len(t) == 3 and t[:2] == '##'])
        )
        vocab_prefix = sorted([t for t in vocab[original_len:] if t[:2] != '##' and len(t) > 1])
        vocab_suffix = sorted([t for t in vocab[original_len:] if t[:2] == '##' and len(t) > 3])

        vocab = vocab[:original_len] + vocab_single_chars + vocab_prefix + vocab_suffix
    else:
        raise ValueError("`mode` has to be one of ('a', 'w').")

    logging.info('Total vocab size:', len(vocab))

    # save the vocabulary to the destination tokenizer path
    with open(original_vocab_path, 'w') as f:
        f.write('\n'.join(vocab))
    logging.info('Saved merged vocabulary to {}'.format(original_vocab_path))


@click.command()
@click.option("--src_txt_glob_path", help="Input files pathname pattern, e.g., /path/to/files/*_abc_*.txt")
@click.option("--num_threads", default=24, help="Number of threads to use.")
@click.option("--vocab_size", default=10_001, help="Output vocabulary size.")
@click.option("--tokenizer_name", help="Valid model id on huggingface.co or path to a directory containing vocab.txt file.")
@click.option("--dest_tokenizer_name", default=uuid.uuid4().hex[:6].upper(), help="Path to output tokenizer directory with merged vocabulary")
@click.option("--mode", default="a", help="""mode="a" appends the extra vocabulary to the end of the original. mode="w" concatenates and sorts the merged vocabulary. In both cases, duplicates will be removed.""")
def main(
    src_txt_glob_path,
    num_threads,
    vocab_size,
    tokenizer_name,
    dest_tokenizer_name,
    mode,
):
    """
    Create a new tokenizer with merged vocabulary from existing BERT tokenizer
    and extra vocabulary extracted from input text files.
    """
    tmp = "."  #tempfile.gettempdir()
    model_prefix = os.path.join(tmp, uuid.uuid4().hex[:6].upper())
    path_wordpiece_vocab = os.path.join(tmp, uuid.uuid4().hex[:6].upper() + '.vocab')
    extract_sentencepiece_vocab(src_txt_glob_path, model_prefix, num_threads, vocab_size)
    convert_vocab_to_wordpiece(model_prefix + ".vocab", path_wordpiece_vocab)
    merge_original_and_extra_vocab(tokenizer_name, dest_tokenizer_name, path_wordpiece_vocab, mode)

    # cleanup temp files
    if os.path.exists(model_prefix + '.model'):
        os.remove(model_prefix + '.model')
    if os.path.exists(model_prefix + '.vocab'):
        os.remove(model_prefix + '.vocab')
    if os.path.exists(path_wordpiece_vocab):
        os.remove(path_wordpiece_vocab)


if __name__ == "__main__":
    main()
