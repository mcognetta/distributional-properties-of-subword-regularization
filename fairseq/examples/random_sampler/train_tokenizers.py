from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Sequence, Whitespace, ByteLevel, WhitespaceSplit
import tokenizers
import maxmatch_dropout

import sentencepiece as spm

# import pynini as pn
# import pywrapfst as fst

import functools

# from collections import Counter

# import pynini_utils
import argparse
import json
import os
import random
from pathlib import Path

import asyncio

MARKER = "﹏"


def build_bpe_tokenizer(path, vocab_size):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        continuing_subword_prefix=MARKER,
        special_tokens=special_tokens,
        show_progress=False,
    )
    tokenizer.train([path], trainer)

    return tokenizer


def build_wp_tokenizer(path, vocab_size, outpath=None, other_paths=[], to_lower=False):
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        continuing_subword_prefix=MARKER,
        special_tokens=special_tokens,
        show_progress=False,
    )
    tokenizer.train([path] + other_paths, trainer)

    return tokenizer


def build_unigram_tokenizer(
    path, vocab_size, outpath=None, other_paths=[], to_lower=False
):
    tokenizer = Tokenizer(Unigram())
    # tokenizer.pre_tokenizer = WhitespaceSplit()
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=False,
    )
    tokenizer.train([path] + other_paths, trainer)
    return tokenizer


def make_dropout(tokenizer, p=0.1):
    tokenizer.model.dropout = p
    return tokenizer


def make_not_dropout(tokenizer):
    tokenizer.model.dropout = 0.0
    return tokenizer


class HfTokenizerWrapper:

    def __init__(self, tokenizer, dropout):
        self._tokenizer = tokenizer
        self._dropout = dropout

        self._tokenizer.model.dropout = self._dropout

    def encode(self, sentence):
        return " ".join(self._tokenizer.encode(sentence).tokens)

    def turn_off_dropout(self):
        self._dropout = 0.0
        self._tokenizer.model.dropout = 0.0

    def turn_on_dropout(self, dropout):
        self._dropout = dropout
        self._tokenizer.model.dropout = dropout

    def decode(self, tokens):
        return " ".join(self._tokenizer.decode(tokens))

    def dump_to_dir(self, path):
        os.makedirs(path, exist_ok=True)
        self._tokenizer.model.dropout = None
        self._tokenizer.save(os.path.join(path, "tokenizer.json"))
        with open(os.path.join(path, "vocab.txt"), "w") as f:
            for i, w in enumerate(self._tokenizer.get_vocab()):
                f.write(f"{w} {i + 1000}\n")

        data = {
            "tokenizer_path": os.path.join(path, "tokenizer.json"),
            "vocab_path": os.path.join(path, "vocab.txt"),
            "tokenizer": self._tokenizer.to_str(),
            "dropout": self._dropout,
        }

        with open(os.path.join(path, "data.json"), "w") as outfile:
            json.dump(data, outfile, indent=4)

    @classmethod
    def read_from_dir(cls, path):
        tokenizer = tokenizers.Tokenizer.from_file(os.path.join(path, "tokenizer.json"))

        with open(os.path.join(path, "data.json")) as fd:
            data = json.load(fd)
        # tokenizer = Tokenizer.from_str(data['tokenizer'])
        dropout = data["dropout"]

        return cls(tokenizer, dropout)


class MaxMatchTokenizerWrapper:

    def __init__(self, tokenizer, dropout):
        self._tokenizer = tokenizer
        self._dropout = dropout

        self._mm_dropout_tokenizer = maxmatch_dropout.MaxMatchTokenizer(
            vocab=self._tokenizer.get_vocab(), midPref=MARKER, headPref=""
        )

    def encode(self, sentence):
        if self._dropout == 0.0:
            return " ".join(self._tokenizer.encode(sentence).tokens)
        else:
            return " ".join(
                self._mm_dropout_tokenizer.tokenize(sentence, p=self._dropout)
            )

    def turn_off_dropout(self):
        self._dropout = 0.0

    def turn_on_dropout(self, dropout):
        self._dropout = dropout

    def decode(self, tokens):
        return " ".join(self._tokenizer.decode(tokens))

    def dump_to_dir(self, path):
        os.makedirs(path, exist_ok=True)
        self._tokenizer.save(os.path.join(path, "tokenizer.json"))
        with open(os.path.join(path, "vocab.txt"), "w") as f:
            for i, w in enumerate(self._tokenizer.get_vocab()):
                f.write(f"{w} {i + 1000}\n")

        data = {
            "tokenizer_path": os.path.join(path, "tokenizer.json"),
            "vocab_path": os.path.join(path, "vocab.txt"),
            "tokenizer": self._tokenizer.to_str(),
            "dropout": self._dropout,
        }

        with open(os.path.join(path, "data.json"), "w") as outfile:
            json.dump(data, outfile, indent=4)

    @classmethod
    def read_from_dir(cls, path):
        tokenizer = tokenizers.Tokenizer.from_file(os.path.join(path, "tokenizer.json"))

        with open(os.path.join(path, "data.json")) as fd:
            data = json.load(fd)
        # tokenizer = Tokenizer.from_str(data['tokenizer'])
        dropout = data["dropout"]

        return cls(tokenizer, dropout)


class UnigramTokenizerWrapper:

    def __init__(self, tokenizer, dropout=0.0):
        self._tokenizer = tokenizer
        self._dropout = dropout

    def encode(self, sentence):
        return " ".join(
            self._tokenizer.encode(
                input=sentence,
                out_type=str,
                enable_sampling=self._dropout != 0.0,
                alpha= 1 - self._dropout,
                nbest_size=-1,
            )
        )

    def turn_off_dropout(self):
        self._dropout = 0.0

    def turn_on_dropout(self, dropout):
        self._dropout = dropout

    def decode(self, tokens):
        return self._tokenizer.decode(tokens)

    def dump_to_dir(self, path):
        os.makedirs(path, exist_ok=True)

        vocab_probs = open(os.path.join(path, "tokenizer.vocab"), 'r')
        with open(os.path.join(path, "vocab.txt"), "w") as f:
            for (i, l) in enumerate(vocab_probs):
                v, _ = l.strip().split()
                if v not in ["<unk>", "<s>", "</s>"]:
                    f.write(f"{v} {i + 1000}\n")
        vocab_probs.close()

        data = {
            "tokenizer_path": os.path.join(path, "tokenizer.model"),
            "vocab_path": os.path.join(path, "vocab.txt"),
            "dropout": self._dropout,
        }

        with open(os.path.join(path, "data.json"), "w") as outfile:
            json.dump(data, outfile, indent=4)

    @classmethod
    def read_from_dir(cls, path):
        tokenizer = spm.SentencePieceProcessor(model_file=os.path.join(path, "tokenizer.model"))
        with open(os.path.join(path, "data.json")) as fd:
            data = json.load(fd)
        # tokenizer = Tokenizer.from_str(data['tokenizer'])
        dropout = data["dropout"]
        return cls(tokenizer, dropout)

class TrieNode:
    def __init__(self):
        self._children = {}
        self._w = None


def build_trie(vocab):
    HEAD = TrieNode()

    for w in vocab:
        cur = HEAD
        for c in w:
            if c not in cur._children:
                cur._children[c] = TrieNode()
            cur = cur._children[c]
        cur._w = w
    return HEAD


class UniformSamplerWrapperPurePython:
    def __init__(self, tokenizer, lexicon, isyms, osyms, dropout):
        self._tokenizer = tokenizer
        self._lexicon = lexicon
        self._isyms = isyms
        self._osyms = osyms
        self._dropout = dropout

        init_vocab = sorted(c for c in self._tokenizer.get_vocab() if MARKER not in c)
        internal_vocab = sorted(c for c in self._tokenizer.get_vocab() if MARKER in c)
        self._init_trie = build_trie(init_vocab)
        self._internal_trie = build_trie(w.replace(MARKER, "") for w in internal_vocab)

        self._M = max(len(c) for c in init_vocab + internal_vocab)

    def dump_to_dir(self, path):
        os.makedirs(path, exist_ok=True)
        self._tokenizer.save(os.path.join(path, "tokenizer.json"))
        # self._lexicon.write(os.path.join(path, "lexicon.fst"))
        # self._isyms.write_text(os.path.join(path, "isyms.txt"))
        # self._osyms.write_text(os.path.join(path, "osyms.txt"))
        with open(os.path.join(path, "vocab.txt"), "w") as f:
            for i, w in enumerate(self._tokenizer.get_vocab()):
                f.write(f"{w} {i + 1000}\n")

        data = {
            "tokenizer_path": os.path.join(path, "tokenizer.json"),
            "vocab_path": os.path.join(path, "vocab.txt"),
            "tokenizer": self._tokenizer.to_str(),
            # "lexicon": os.path.join(path, "lexicon.fst"),
            # "isyms": os.path.join(path, "isyms.txt"),
            # "osyms": os.path.join(path, "osyms.txt"),
            "dropout": self._dropout,
        }

        with open(os.path.join(path, "data.json"), "w") as outfile:
            json.dump(data, outfile, indent=4)

    def turn_off_dropout(self):
        self._dropout = 0.0

    def turn_on_dropout(self, dropout):
        self._dropout = dropout

    def _recur(self, state, skeleton, cache):
        if len(state._children) == 0:
            cache[state] = 1
            return 1
        if state not in cache:
            cache[state] = sum(
                self._recur(n, skeleton, cache) for n in state._children.values()
            )
        return cache[state]

    @functools.lru_cache(maxsize=20000)
    def _encode_word(self, word):
        skeleton = [TrieNode() for _ in range(len(word) + 1)]

        for i, c in enumerate(word):
            cur_trie = self._internal_trie if i > 0 else self._init_trie
            cur_skeleton = skeleton[i]
            for j in range(i, min(len(word), i + self._M + 1)):
                if word[j] in cur_trie._children:
                    if cur_trie._children[word[j]]._w:
                        cur_skeleton._children[cur_trie._children[word[j]]._w] = (
                            skeleton[j + 1]
                        )
                    cur_trie = cur_trie._children[word[j]]
                else:
                    break

        cache = {}

        self._recur(skeleton[0], skeleton, cache)

        return skeleton, cache

    def _sample(self, word):
        if len(word) <= 1:
            return word
        skeleton, cache = self._encode_word(word)
        path_index = random.randint(0, cache[skeleton[0]] - 1)
        path = []
        cur = skeleton[0]
        at_internal_state = False
        while len(cur._children) > 0:

            running_sum = 0
            arcs = list(cur._children.items())
            if len(arcs) == 1:
                path.append((MARKER if at_internal_state else "") + arcs[0][0])
                cur = arcs[0][1]
            else:
                next_arc = None
                running_sum = 0
                for i, (label, nextstate) in enumerate(arcs):

                    if running_sum + cache[nextstate] <= path_index:
                        running_sum += cache[nextstate]
                    else:
                        next_arc = arcs[i]
                        break
                if not next_arc:
                    next_arc = arcs[-1]
                path.append((MARKER if at_internal_state else "") + label)
                cur = nextstate
                path_index -= running_sum
            at_internal_state = True
        return " ".join(path)

    def encode(self, sentence):
        if self._dropout == 0.0:
            return " ".join(self._tokenizer.encode(sentence).tokens)
        else:
            s = sentence.split(" ")
            bpe = [" ".join(enc.tokens) for enc in self._tokenizer.encode_batch(s)]
            return " ".join(
                b if random.random() > self._dropout else self._sample(w)
                for (b, w) in zip(bpe, s)
            )
    
    def decode(self, tokens):
        return " ".join(self._tokenizer.decode(tokens))

    @classmethod
    def read_from_dir(cls, path):
        with open(os.path.join(path, "data.json")) as fd:
            data = json.load(fd)
        tokenizer = Tokenizer.from_str(data["tokenizer"])

        dropout = data["dropout"]

        return cls(tokenizer, None, None, None, dropout)

    @classmethod
    def build_from_tokenizer(cls, tokenizer, dropout=0.0):
        return cls(tokenizer, None, None, None, dropout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer-type",
        type=str,
        help="the tokenizer type (bpe, maxmatch, bpe-uniform, maxmatch-uniform)",
        required=True,
    )
    parser.add_argument(
        "--vocab-size", type=int, help="size of the vocabluary", default=10_000
    )
    parser.add_argument("--dropout", type=float, help="dropout p", default=0.0)
    parser.add_argument(
        "--input-data-path", type=str, help="path to training data", required=True
    )
    parser.add_argument(
        "--tokenizer-output-dir",
        type=str,
        help="dir to store tokenizer information",
        required=True,
    )

    args = parser.parse_args()

    if args.tokenizer_type == "bpe":
        tokenizer = build_bpe_tokenizer(args.input_data_path, args.vocab_size)
        wrapped = HfTokenizerWrapper(tokenizer, args.dropout)
        wrapped.dump_to_dir(args.tokenizer_output_dir)
    elif args.tokenizer_type in ["bpe_uniform_python", "bpe-uniform-python"]:
        tokenizer = build_bpe_tokenizer(args.input_data_path, args.vocab_size)
        wrapped = UniformSamplerWrapperPurePython.build_from_tokenizer(
            tokenizer, args.dropout
        )
        wrapped.dump_to_dir(args.tokenizer_output_dir)
    elif args.tokenizer_type in ["wordpiece", "maxmatch"]:
        tokenizer = build_wp_tokenizer(args.input_data_path, args.vocab_size)
        wrapped = MaxMatchTokenizerWrapper(tokenizer, args.dropout)
        wrapped.dump_to_dir(args.tokenizer_output_dir)
    elif args.tokenizer_type in [
        "wordpiece_uniform",
        "maxmatch_uniform",
        "wordpiece_uniform_python",
        "maxmatch_uniform_python",
    ]:
        tokenizer = build_wp_tokenizer(args.input_data_path, args.vocab_size)
        wrapped = UniformSamplerWrapperPurePython.build_from_tokenizer(
            tokenizer, args.dropout
        )
        wrapped.dump_to_dir(args.tokenizer_output_dir)
    elif args.tokenizer_type in ["unigram"]:
        tokenizer = spm.SentencePieceTrainer.train(
            input=args.input_data_path,
            model_prefix=os.path.join(args.tokenizer_output_dir, "tokenizer"),
            vocab_size=args.vocab_size,
            character_coverage=1.0,
            model_type="unigram",
        )
        wrapped = UnigramTokenizerWrapper(tokenizer)
        wrapped.dump_to_dir(args.tokenizer_output_dir)
    else:
        raise ValueError(f"tokenizer type {args.tokenizer_type} not yet supported")
