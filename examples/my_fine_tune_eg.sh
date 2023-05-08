task=$1
mkdir -p ./result/$KMER/$task
export KMER=6
export DATA_PATH=./sample_data/ft/$KMER/$task
export MODEL_PATH=./ft/$KMER
export OUTPUT_PATH=./result/$KMER/$task


# the logging step has been change from 100 to 1000
# NOTE THAT THE LEARNING RATE HAS BEEN CHANGED!
# the last line is changed but probably does not matter.
python my_run_finetune.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_train \
    --do_eval \
    --data_dir $DATA_PATH \
    --max_seq_length 501 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 5e-5 \
    --num_train_epochs 5.0 \
    --output_dir $OUTPUT_PATH \
    --evaluate_during_training \
    --logging_steps 5000 \
    --save_steps 4000 \
    --warmup_percent 0.1 \
    --hidden_dropout_prob 0.1 \
    --overwrite_output \
    --weight_decay 0.01 \
    --n_process 8
