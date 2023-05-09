#!/bin/bash
## Job Name
#SBATCH --job-name=ar-template-db
## Allocation Definition: The account and partition options should be the same except in a few cases (e.g. ckpt queue and genpool queue).
#SBATCH --nodes=1
#SBATCH --account=ark
#SBATCH --partition=gpu-2080ti
#SBATCH --gpus=1 ## Make sure to change gpu count here and in below constant!
#SBATCH --cpus-per-gpu=4
## Number of cores per node
#SBATCH --ntasks-per-node=1
## Walltime (3 hours). Do not specify a walltime substantially more than your job needs.
#SBATCH --time=00:07:00
## Memory per node. It is important to specify the memory since the default memory is very small.
## For mox, --mem may be more than 100G depending on the memory of your nodes.
## For ikt, --mem may be 58G or more depending on the memory of your nodes.
## See above section on "Specifying memory" for choices for --mem.
#SBATCH --mem=32G
## Specify the working directory for this job
#SBATCH --chdir=/gscratch/ark/risuka/wikitext/slurm_output
##turn on e-mail notification
#SBATCH --mail-type=ALL
#SBATCH --mail-user=risuka@cs.washington.edu
## export all your environment variables to the batch job session
#SBATCH --export=all
#SBATCH --output=/gscratch/ark/risuka/wikitext/slurm_output/%j
## %j is job number

## The equation (max tokens / tokens per sample) * update freq * # gpus = batch size
## Andrey usage: NUM_GPUS will vary as available, and update_freq needs to compensate

## TOKENS_PER_SAMPLE specifies size of input chunk
## MAX_TOKENS specifies how many tokens on each gpu are loaded (the number of gpu changes how many chunks per batch, 1:2, 8:16 )
    # Try fewer than tokens per sample, observe error; try not int division, observe error
## UPDATE_FREQ indicates how many (#) forward and backward passes per 1 gradient update (affects amount of data we see per gradient update)
## NUM_GPUS as above

## MAX_TOKENS=1024  should change based on GPU I am using; once hop onto new node, change everything in formula for bsz
## Maximize max tokens until memory error

# TOKENS_PER_SAMPLE=512 # Important to keep constant for all arch train and inference, allows models to be comparable! 

NUM_GPUS=1
DEVICE=0
# NUM_GPUS=2
# DEVICE=0,1
DATA=wikitext-103-ten-percent

export XDG_CACHE_HOME="/gscratch/ark/risuka/.cache"
export CUDA_VISIBLE_DEVICES=$DEVICE
export MKL_SERVICE_FORCE_INTEL=1

if [[ ${SBATCH_PARTITION} == "gpu-2080ti" || ${SBATCH_PARTITION} == "" ]]; then
    FRAGS_PER_GPU=4
elif [[ ${SBATCH_PARTITION} == "gpu-titan" || ${SBATCH_PARTITION} == "gpu-rtx6k" ]]; then
    FRAGS_PER_GPU=16
elif [[ ${SBATCH_PARTITION} == "gpu-a40" ]]; then
    FRAGS_PER_GPU=32
else
    echo "Wrong partition: $1"
    exit 0
fi

if [[ ${DATA} == "wikitext-103-full" ]]; then
    MAX_UPDATE=300000 ## TODO tune these
    WARMUP_UPDATES=6000
elif [[ ${DATA} == "wikitext-103-ten-percent" ]]; then
    MAX_UPDATE=50000
    WARMUP_UPDATES=4000
else
    echo "Unexpected data: ${DATA}"
    exit 0
fi

# MAX_TOKENS=$(python -c "print(${FRAGS_PER_GPU}*1024)")
# MAX_TOKENS=1024 # TODO ask Hao, what is the effect of me hardcoding this way? What are downsides, -> what are other ways to fix bsz and tokens_per_sample correctly?
UPDATE_FREQ=$(python -c "print(int(32//${FRAGS_PER_GPU}//${NUM_GPUS}))")
# UPDATE_FREQ=$(python -c "print(int(32//${FRAGS_PER_GPU}//${NUM_GPUS}))"/2)

BSZ=16
TOKENS_PER_SAMPLE=512 # Constant to be comparable?
MAX_TOKENS=$(python -c "print(int(${TOKENS_PER_SAMPLE}/${UPDATE_FREQ}/${NUM_GPUS}*${BSZ}))") # This adjust max tokens based on num GPUs to keep BSZ and tokens per sample constant. TODO there's a case where max tokens is calculated as less than 1, not int; would need to adjust update_freq in this case.

## Testing failure case when MAX_TOKENS = TOKENS_PER_SAMPLE
# MAX_TOKENS=512

## Testing failure case when MAX_TOKENS is not cleanly divisible by TOKENS_PER_SAMPLE
MAX_TOKENS=$(python -c "print(int(${MAX_TOKENS}) + 1)")

# TOKENS_PER_SAMPLE=$(python -c "print(int(${UPDATE_FREQ}*${MAX_TOKENS}*${NUM_GPUS}//${BSZ}))")
# TOKENS_PER_SAMPLE=$(python -c "print(min(${TOKENS_PER_SAMPLE}, 512))") # TODO might not be the preferred way to correct

# TOKENS_PER_SAMPLE is upper bounded by GPU size, can always use something smaller 
# For example, if we can use fewer gpus with a large tokens_per_sample, we can run more experiments than using more gpus with a small tokens_per_sample

SEED=42
LR=0.0005 #1e-07 ##7e-4
DROPOUT=0.1 ##0.3
WEIGHT_DECAY=0.01 #0

echo "${DATA} Templates Baseline"
echo "$SBATCH_ACCOUNT $SBATCH_PARTITION"
echo "lr=$LR warmup=$WARMUP_UPDATES"
echo "drop=$DROPOUT weight_decay=$WEIGHT_DECAY"
echo "seed=$SEED"
echo "BSZ=$BSZ TOKENS_PER_SAMPLE=$TOKENS_PER_SAMPLE NUM_GPUS=$NUM_GPUS MAX_TOKENS=$MAX_TOKENS UPDATE_FREQ=$UPDATE_FREQ"

DATA_PATH=/gscratch/ark/risuka/wikitext/data-bin/${DATA}
RUN_NAME=check_softmaxed_coefficients ## Need to update this each time GPU setting changes, else remembers old setting
SAVE_DIR=/gscratch/ark/risuka/wikitext/checkpoints/${RUN_NAME}
TBLOG_DIR=/gscratch/ark/risuka/wikitext/tblogs/${RUN_NAME}

if [ -d $SAVE_DIR ]; then
    echo "Dir $SAVE_DIR already exists; skipping"
else
    mkdir -p $SAVE_DIR
fi

/mmfs1/gscratch/ark/risuka/miniconda/envs/dev/bin/python \
/gscratch/ark/risuka/efftr/fairseq/fairseq_cli/train.py \
    ${DATA_PATH} \
    --task language_modeling \
    --arch templates_lm \
    --update-freq ${UPDATE_FREQ} \
    --save-dir ${SAVE_DIR} \
    --no-epoch-checkpoints \
    --optimizer adam \
    --seed 42 \
    --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --warmup-init-lr 1e-07 \
    --max-tokens ${MAX_TOKENS} \
    --tokens-per-sample ${TOKENS_PER_SAMPLE} \
    --lr ${LR} \
    --weight-decay ${WEIGHT_DECAY} \
    --dropout ${DROPOUT} \
    --max-update ${MAX_UPDATE} \
    --warmup-updates ${WARMUP_UPDATES} \
    --clip-norm 0. \
    --fp16 \
    --find-unused-parameters \
    --max-epoch 300 \
    --tensorboard-logdir ${TBLOG_DIR}

# --tokens-per-sample ${TOKENS_PER_SAMPLE} \
# --batch-size ${BSZ} \

# /mmfs1/gscratch/ark/hao/conda/envs/prior/bin/python \
# /gscratch/ark/hao/abc/fairseq_cli/train.py \
#     ${DATA_PATH} \
#     --arch transformer_wmt_en_de \
#     --update-freq ${UPDATE_FREQ} \
#     --save-dir ${SAVE_DIR} \
#     --no-epoch-checkpoints \
#     --no-last-checkpoints \
#     --keep-best-checkpoints 5 \
#     --activation-fn "gelu" \
#     --share-all-embeddings \
#     --optimizer adam \
#     --seed 42 \
#     --adam-betas '(0.9, 0.98)' \
#     --lr-scheduler inverse_sqrt \
#     --criterion label_smoothed_cross_entropy \
#     --label-smoothing 0.1 \
#     --warmup-init-lr 1e-07 \
#     --max-tokens ${MAX_TOKENS} \
#     --lr ${LR} \
#     --weight-decay ${WEIGHT_DECAY} \
#     --dropout ${DROPOUT} \
#     --max-update ${MAX_UPDATE} \
#     --warmup-updates ${WARMUP_UPDATES} \
#     --clip-norm 0. \
#     --fp16 \
#     --fp16-init-scale 16 \
#     --fp16-scale-window 1024 \
#     --ddp-backend=no_c10d \
#     --find-unused-parameters \
#     --eval-bleu \
#     --eval-bleu-args '{"beam": 1, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-bleu-detok moses \
#     --eval-bleu-remove-bpe \
#     --best-checkpoint-metric bleu \
#     --maximize-best-checkpoint-metric