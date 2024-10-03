#! /bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodelist=gcn[7-72]
##SBATCH --exclude=gcn1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2-10

export PATH="/home/dwu18/anaconda3/bin:$PATH"
source activate llama_factory

N_TRAINING=100
BASE_MODEL="meta-llama/Llama-2-7b-hf"
MODEL_NAME="Llama-2-7b-hf"

# BASE_MODEL="Unbabel/TowerInstruct-7B-v0.1"
# MODEL_NAME="TowerInstruct-7B-v0.1"

PORT=12345
TEMPLATE="This_sentence_:_\"*sent_0*\"_means_in_one_word:\""

:<<!
# Model training
WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=${PORT} scripts/ft_llm.py \
    --base_model $BASE_MODEL \
    --data_path crosslingual_data/NTREX-en-prompts-en2x.${N_TRAINING}.csv \
    --batch_size 256 \
    --micro_batch_size 4 \
    --num_epochs 1 \
    --learning_rate 5e-4 \
    --cutoff_len 32 \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --output_dir lora_dir/llama2-${N_TRAINING} --is_sentemb \
    --save_steps 20 ----load_kbit 16
!

# Model evaluation
python eval.py \
    --model_name_or_path $BASE_MODEL \
    --mode test --mask_embedding_sentence \
    --mask_embedding_sentence_template $TEMPLATE

# Model evaluation 100 samples
python eval.py \
    --model_name_or_path $BASE_MODEL \
    --lora_weight /home/dwu18/projects/multilingual_LLMs/lora_dir/${MODEL_NAME}/en-prompts-128-10-en2x-100 \
    --mode test --mask_embedding_sentence \
    --mask_embedding_sentence_template $TEMPLATE

# Model evaluation 1000 samples
python eval.py \
    --model_name_or_path $BASE_MODEL \
    --lora_weight /home/dwu18/projects/multilingual_LLMs/lora_dir/${MODEL_NAME}/en-prompts-128-3-en2x-1000 \
    --mode test --mask_embedding_sentence \
    --mask_embedding_sentence_template $TEMPLATE
