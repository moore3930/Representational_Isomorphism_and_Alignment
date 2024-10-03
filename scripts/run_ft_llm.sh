EPOCH=('3')
CUTOFF=('128')
SETTING=('en-prompts')
DATA_TYPE=('en2x')

BASE_MODEL="meta-llama/Llama-2-7b-hf"
MODEL="Llama-2-7b-hf"
PORT=1234

for i1 in "${!EPOCH[@]}"; do
    for i2 in "${!CUTOFF[@]}"; do
        for i3 in "${!SETTING[@]}"; do
	    for i4 in "${!DATA_TYPE[@]}"; do
	        sbatch ft_llm.sh ${EPOCH[i1]} ${CUTOFF[i2]} ${SETTING[i3]} ${DATA_TYPE[i4]} ${BASE_MODEL} ${MODEL} ${PORT}
	        PORT=$((PORT + 1))
	    done
        done
    done
done
