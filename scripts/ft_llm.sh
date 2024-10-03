#! /bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
##SBATCH --nodelist=ilps-cn115
#SBATCH --exclude=ilps-cn111,ilps-cn101,ilps-cn102,ilps-cn103,ilps-cn104,ilps-cn105,ilps-cn106,ilps-cn107,ilps-cn108,ilps-cn109,ilps-cn110,ilps-cn112,ilps-cn113,ilps-cn114,ilps-cn115
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=7-10

#SBATCH -o /home/dwu/workplace/logs/multilingual_LLMs/out.ft_llm.%j.o
#SBATCH -e /home/dwu/workplace/logs/multilingual_LLMs/out.ft_llm.%j.e

source activate llm_39

# EPOCH=1
# CUTOFF=128
# SETTING=en-prompts

# BASE_MODEL="meta-llama/Llama-2-7b-chat-hf"
# MODEL="Llama-2-7b-chat-hf"

# BASE_MODEL="meta-llama/Llama-2-7b-hf"
# MODEL="Llama-2-7b-hf"

EPOCH=$1
CUTOFF=$2
SETTING=$3
DATA_TYPE=$4
BASE_MODEL=$5
MODEL=$6
PORT=$7

echo $EPOCH
echo $CUTOFF
echo $SETTING
echo $DATA_TYPE
echo $BASE_MODEL
echo $MODEL
echo $PORT

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=${PORT} scripts/ft_llm.py \
    --base_model $BASE_MODEL \
    --data_path crosslingual_data/NTREX/NTREX-${SETTING}-${DATA_TYPE}.3lang.1000.csv \
    --batch_size 128 \
    --micro_batch_size 32 \
    --num_epochs ${EPOCH} \
    --learning_rate 5e-4 \
    --cutoff_len ${CUTOFF} \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --output_dir lora_dir/${MODEL}/${SETTING}-${CUTOFF}-${EPOCH}-${DATA_TYPE}-3lang-1000sample-debug --is_sentemb \
    --save_steps 5 --load_kbit 16

