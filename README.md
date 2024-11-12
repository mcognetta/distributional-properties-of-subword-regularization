# Distributional Properties of Subword Regularization

This is the source code for the paper [`Distributional Properties of Subword Regularization`](https://aclanthology.org/2024.emnlp-main.600/), presented at EMNLP2024.

# How to use

NOTE: We only include EN<->DE here, as we don't have an easy way to distribute the other datasets (though they are publically available). Adding a new dataset just requires swapping out the corpus data. All other code/steps remain the same.

- Clone directory
- `pip install sentencepiece tokenizers sacrebleu sacremoses`
- In `fairseq`
    - `pip install --editable ./`
- In `fairseq/examples/random_sampler`
    - This is an EN<->DE model
    - Prepare data: `bash prepare_data.sh`
    - Run a training run: `bash train_de_en.sh --experiment-name test --tokenizer-type bpe_uniform_python --src-dropout 0.1 --seed 0`
    - The output will be in `fairseq/<langpair>_experiments/` (see below for the directory name)


- Invocation options:
    - `bash train_de_en.sh --experiment-name <EXPERIMENT_NAME> --tokenizer-type <TOKENIZER_TYPE> --src-bpe-tokens <STOKENS> --tgt-bpe-tokens <TTOKENS> --src-dropout <SDROPOUT> --target-dropout <TDROPOUT> --seed <SEED>`
    - `<EXPERIMENT_NAME>` is any string identifier
    - `<TOKENIZER_TYPE>` is one of `bpe,bpe-uniform-python,maxmatch,maxmatch-uniform,unigram`
    - `<STOKENS>` and `<TTOKENS>` are the size of the source and target tokens (default 10k each)
    - `<SDROPOUT>` and `<TDROPOUT>` are dropout parameters for the tokenizer they should be between 0.0 and 1.0 (default is 0.0, which means no dropout)
        - for unigram, the values are reversed from the original implementation of unigram (i.e., `0.0` equivalent to `alpha = 1.0`, and vice-versa)
    - `<SEED>` is the random seed for training
    - The output folder name is `<EXPERIMENT_NAME>_VOCAB_<STOKENS>_<TTOKENS>_type_<TOKENIZER_TYPE>_dropout_<SDROPOUT>_<TDROPOUT>_seed_<SEED>.<LANG_PAIR>` in the experiment outputs directory
 
# Poster
![poster](https://github.com/user-attachments/assets/7bd73e72-fc9e-488b-8faa-f9e1bbcd3c3e)
