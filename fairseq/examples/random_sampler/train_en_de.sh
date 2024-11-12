#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt
#BPE_TOKENS=10000
SRC_BPE_TOKENS=10000
TGT_BPE_TOKENS=10000
SRC_DROPOUT=0.0
TGT_DROPOUT=0.0
TOKENIZER_TYPE="bpe"
SEED=0
DEVICE=0

EXPERIMENT_PREFIX="experiment"

while [[ "$#" -gt 0 ]]
do case $1 in
    --src-bpe-tokens) SRC_BPE_TOKENS=$2
    shift;;
    --tgt-bpe-tokens) TGT_BPE_TOKENS=$2
    shift;;
    --tokenizer-type) TOKENIZER_TYPE=$2
    shift;;
    --src-dropout) SRC_DROPOUT=$2
    shift;;
    --tgt-dropout) TGT_DROPOUT=$2
    shift;;
    --seed) SEED=$2
    shift;;
    --device) DEVICE=$2
    shift;;
    --experiment-name) EXPERIMENT_PREFIX="$2"
    shift;;
    *) echo "Unknown parameter passed: $1"
    exit 1;;
esac
shift
done
echo "========= PARAMETERS =========== "
echo -e "SRC_TOKENS $SRC_BPE_TOKENS \nTGT_TOKENS $TGT_BPE_TOKENS \nTOKENIZER_TYPE ${TOKENIZER_TYPE} \nSRC_DROPOUT $SRC_DROPOUT \nTGT_DROPOUT $TGT_DROPOUT \nSEED $SEED \nDEVICE $DEVICE \nNAME $EXPERIMENT_PREFIX\n"
echo "========= PARAMETERS =========== "


src=en
tgt=de
lang=en-de

EXPERIMENT_NAME="${EXPERIMENT_PREFIX}_BPE_${SRC_BPE_TOKENS}_${TGT_BPE_TOKENS}_type_${TOKENIZER_TYPE}_dropout_${SRC_DROPOUT}_${TGT_DROPOUT}_seed_${SEED}.${lang}"

prep=experiments/$EXPERIMENT_NAME
tmp=$prep/tmp
orig=orig


if [ -d "../../en_de_experiments/${EXPERIMENT_NAME}" ]
then
    echo "${EXPERIMENT_NAME} already done, SKIPPING"
    exit 0
fi

mkdir -p $prep

mkdir -p data-bin/$EXPERIMENT_NAME

BPE_CODE=$prep/code
BPE_VOCAB=$prep/vocab

mkdir -p $prep/${src}_tokenizer
mkdir -p $prep/${tgt}_tokenizer

echo "learn_BPE for src: $src"
# python3 $BPEROOT/learn_joint_bpe_and_vocab.py --input $orig/train.$src -s $SRC_BPE_TOKENS -t -o $BPE_CODE.$src --write-vocabulary $BPE_VOCAB.$src
python3 train_tokenizers.py --tokenizer-type ${TOKENIZER_TYPE} --input-data-path $orig/train.$src --tokenizer-output-dir $prep/${src}_tokenizer


echo "learn_BPE for tgt: $tgt"
# python3 $BPEROOT/learn_joint_bpe_and_vocab.py --input $orig/train.$tgt -s $TGT_BPE_TOKENS -t -o $BPE_CODE.$tgt --write-vocabulary $BPE_VOCAB.$tgt
python3 train_tokenizers.py --tokenizer-type ${TOKENIZER_TYPE} --input-data-path $orig/train.$tgt --tokenizer-output-dir $prep/${tgt}_tokenizer

for f in train valid test; do
    echo "tokenize ($src) to ${f}.${src}..."
    python3 tokenize_corpus.py --tokenizer-type ${TOKENIZER_TYPE} --tokenizer-path $prep/${src}_tokenizer --corpus-path $orig/$f.$src --tokenized-corpus-output-path $prep/$f.$src
    cp $prep/$f.$src data-bin/$EXPERIMENT_NAME/$f.$src

    echo "tokenize ($tgt) to ${f}.${tgt}..."
    python3 tokenize_corpus.py --tokenizer-type ${TOKENIZER_TYPE} --tokenizer-path $prep/${tgt}_tokenizer --corpus-path $orig/$f.$tgt --tokenized-corpus-output-path $prep/$f.$tgt
    cp $prep/$f.$tgt data-bin/$EXPERIMENT_NAME/$f.$tgt
done

cd ../..

TOKENIZERS_PARALLELISM=true

TEXT=examples/random_sampler/experiments/$EXPERIMENT_NAME
fairseq-preprocess --source-lang $src --target-lang $tgt \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir examples/random_sampler/data-bin/$EXPERIMENT_NAME \
    --workers 8 \
    --srcdict $TEXT/${src}_tokenizer/vocab.txt \
    --tgtdict $TEXT/${tgt}_tokenizer/vocab.txt

# cp $orig/train.$src examples/random_sampler/data-bin/$EXPERIMENT_NAME/train.raw.$src
# cp $orig/train.$tgt examples/random_sampler/data-bin/$EXPERIMENT_NAME/train.raw.$tgt

mkdir -p examples/random_sampler/data-bin/$EXPERIMENT_NAME/${src}_tokenizer/
mkdir -p examples/random_sampler/data-bin/$EXPERIMENT_NAME/${tgt}_tokenizer/

cp ${TEXT}/${src}_tokenizer/* examples/random_sampler/data-bin/$EXPERIMENT_NAME/${src}_tokenizer/
cp ${TEXT}/${tgt}_tokenizer/* examples/random_sampler/data-bin/$EXPERIMENT_NAME/${tgt}_tokenizer/

# cp ${TEXT}/${src}_tokenizer/vocab.txt examples/random_sampler/data-bin/$EXPERIMENT_NAME/${src}_tokenizer/vocab.txt
# cp ${TEXT}/${tgt}_tokenizer/vocab.txt examples/random_sampler/data-bin/$EXPERIMENT_NAME/${tgt}_tokenizer/vocab.txt

# sed -i -r 's/(@@ )|(@@ ?$)//g' examples/random_sampler/data-bin/$EXPERIMENT_NAME/train.raw.$src
# sed -i -r 's/(@@ )|(@@ ?$)//g' examples/random_sampler/data-bin/$EXPERIMENT_NAME/train.raw.$tgt

mkdir -p en_de_experiments/${EXPERIMENT_NAME}/

CUDA_VISIBLE_DEVICES=$DEVICE nohup fairseq-train  examples/random_sampler/data-bin/$EXPERIMENT_NAME \
                                            --arch transformer_iwslt_de_en \
                                            --share-decoder-input-output-embed \
                                            --optimizer adam --adam-betas '(0.9, 0.98)' \
                                            --clip-norm 0.0 \
                                            --lr 5e-4 \
                                            --lr-scheduler inverse_sqrt \
                                            --warmup-updates 4000 \
                                            --dropout 0.3 \
                                            --weight-decay 0.0001 \
                                            --criterion label_smoothed_cross_entropy \
                                            --label-smoothing 0.1 \
                                            --max-tokens 8192 \
                                            --eval-bleu  \
                                            --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
                                            --eval-bleu-detok moses \
                                            --eval-bleu-remove-bpe \
                                            --eval-bleu-print-samples \
                                            --best-checkpoint-metric bleu \
                                            --maximize-best-checkpoint-metric \
                                            --patience 8  \
                                            --validate-after-updates 3000 \
                                            --save-dir "en_de_experiments/${EXPERIMENT_NAME}" \
                                            --source-lang=$src \
                                            --target-lang=$tgt \
                                            --seed $SEED \
                                            --task "translation-with-subword-regularization" \
                                            --src-dropout $SRC_DROPOUT \
                                            --tgt-dropout $TGT_DROPOUT \
                                            --tokenizer-config-path "examples/random_sampler/data-bin/$EXPERIMENT_NAME/" \
                                            --tokenizer-type ${TOKENIZER_TYPE} \
                                            --raw-data-path "examples/random_sampler/orig" \
                                            --no-epoch-checkpoints > en_de_experiments/${EXPERIMENT_NAME}/$EXPERIMENT_NAME.log
                                            


CUDA_VISIBLE_DEVICES=$DEVICE nohup fairseq-generate examples/random_sampler/data-bin/$EXPERIMENT_NAME \
                                        --path en_de_experiments/${EXPERIMENT_NAME}/checkpoint_best.pt \
                                        --batch-size 128 \
                                        --beam 5 \
                                        --max-len-a 1.2 \
                                        --max-len-b 10 \
                                        --remove-bpe > en_de_experiments/${EXPERIMENT_NAME}/bleu_unprocessed.log



cd en_de_experiments/${EXPERIMENT_NAME}

grep ^H bleu_unprocessed.log | cut -f3- > gen.out.sys
grep ^T bleu_unprocessed.log | cut -f2- > gen.out.ref
cat gen.out.sys | sacremoses -l $tgt detokenize  > gen.out.sys.detok
cat gen.out.ref | sacremoses -l $tgt detokenize  > gen.out.ref.detok
sacrebleu gen.out.ref.detok -i gen.out.sys.detok -m bleu -b -w 4 > BLEU.txt
sacrebleu gen.out.ref.detok -i gen.out.sys.detok -m chrf -b > CHRF.txt
