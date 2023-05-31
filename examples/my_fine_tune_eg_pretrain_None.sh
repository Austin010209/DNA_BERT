task=$1
pretrain_ind=$2
data_preprocessing=$3
# data_preprocessing: None, seq, label, both

export KMER=6
cd /home/austin29/projects/ctb-liyue/austin29/ATAC/main_method/DNA_BERT/examples/
mkdir -p ./result
mkdir -p ./result/$KMER
mkdir -p ./result/$KMER/pretrain_$pretrain_ind/
mkdir -p ./result/$KMER/pretrain_$pretrain_ind/$data_preprocessing
mkdir -p ./result/$KMER/pretrain_$pretrain_ind/$data_preprocessing/$task

export DATA_PATH=./sample_data/ft/$KMER/$data_preprocessing/$task
export MODEL_PATH=./pretrained_model/$KMER
export OUTPUT_PATH=./result/$KMER/pretrain_$pretrain_ind/$data_preprocessing/$task


# RECHECK THE PARAMETERS



# the logging step has been change from 100 to 1000
# NOTE THAT THE LEARNING RATE HAS BEEN CHANGED!
# the last line is changed but probably does not matter.

# --tokenizer_name=dna$KMER \
python my_run_finetune.py \
    --model_type dna \
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
    --n_process 8 \
    --pretrain $pretrain_ind
