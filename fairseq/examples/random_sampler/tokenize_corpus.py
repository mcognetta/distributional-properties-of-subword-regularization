# from train_tokenizers import UniformSamplerWrapper, HfTokenizerWrapper, UniformSamplerWrapperPurePython
from train_tokenizers import (
    HfTokenizerWrapper,
    UniformSamplerWrapperPurePython,
    MaxMatchTokenizerWrapper,
    UnigramTokenizerWrapper,
)

# import pynini as pn
# import pywrapfst as fst

# import pynini_utils
import argparse, os
from pathlib import Path


def tokenize_corpus(wrapped_tokenizer, corpus_path, out_path):
    os.makedirs(Path(out_path).parent, exist_ok=True)
    corpus = open(corpus_path, "r")
    tokenized = open(out_path, "w")
    for l in corpus:
        tokenized.write(wrapped_tokenizer.encode(l.strip()).strip() + "\n")
    corpus.close()
    tokenized.close()


def load_and_tokenize(tokenizer_type, tokenizer_path, corpus_path, out_path, dropout):
    if tokenizer_type == "bpe":
        tokenizer = HfTokenizerWrapper.read_from_dir(tokenizer_path)
    # elif tokenizer_type in ["bpe_uniform", "bpe-uniform"]:
    #     tokenizer = UniformSamplerWrapper.read_from_dir(tokenizer_path)
    elif tokenizer_type in ["bpe_uniform_python", "bpe-uniform-python"]:
        tokenizer = UniformSamplerWrapperPurePython.read_from_dir(tokenizer_path)
    elif tokenizer_type in ["wordpiece", "maxmatch"]:
        tokenizer = MaxMatchTokenizerWrapper.read_from_dir(tokenizer_path)
    elif tokenizer_type in [
        "wordpiece_uniform",
        "maxmatch_uniform",
        "wordpiece_uniform_python",
        "maxmatch_uniform_python",
    ]:
        tokenizer = UniformSamplerWrapperPurePython.read_from_dir(tokenizer_path)
    elif tokenizer_type in ["unigram"]:
        tokenizer = UnigramTokenizerWrapper.read_from_dir(tokenizer_path)
    else:
        raise ValueError(f"tokenizer type {tokenizer_type} not yet supported")

    tokenizer.turn_on_dropout(dropout)
    tokenize_corpus(tokenizer, corpus_path, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer-type",
        type=str,
        help="the tokenizer type (bpe, maxmatch, bpe-uniform, maxmatch-uniform)",
        required=True,
    )
    parser.add_argument(
        "--tokenizer-path", type=str, help="path to the tokenizer data", required=True
    )
    parser.add_argument("--corpus-path", type=str, help="path to corpus", required=True)
    parser.add_argument(
        "--tokenized-corpus-output-path",
        type=str,
        help="path to store the tokenized corpus",
        required=True,
    )
    parser.add_argument("--dropout", type=float, help="dropout p", default=0.0)

    args = parser.parse_args()

    # if args.tokenizer_type == "bpe":
    #     tokenizer = HfTokenizerWrapper.read_from_dir(args.tokenizer_path)
    # elif args.tokenizer_type == "bpe-uniform":
    #     tokenizer = UniformSamplerWrapper.read_from_dir(args.tokenizer_path)
    # else:
    #     raise ValueError(f"tokenizer type {args.tokenizer_type} not yet supported")

    # tokenizer.turn_on_dropout(args.dropout)
    # tokenize_corpus(tokenizer, args.corpus_path, args.tokenized_corpus_output_path)

    load_and_tokenize(
        args.tokenizer_type,
        args.tokenizer_path,
        args.corpus_path,
        args.tokenized_corpus_output_path,
        args.dropout,
    )
