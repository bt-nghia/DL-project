export MODEL_NAME=roberta-large
export SEED=$1
export DATASET=$2
export LEARNING_RATE=2e-5
export EPOCH=7
export LR_SCHEDULER_TYPE=linear
export OUTPUT_DIR=$DATASET/glue/glue_$MODEL_NAME\_lr.$LEARNING_RATE\_$LR_SCHEDULER_TYPE\_epoch.$EPOCH\_seed.$SEED
export CACHE_DIR=temp\_dir

python3 run_glue_roberta_large.py \
--model_name_or_path $MODEL_NAME  \
--task_name $DATASET \
--do_train \
--do_eval \
--do_predict \
--learning_rate $LEARNING_RATE \
--num_train_epochs $EPOCH  \
--max_seq_length 512  \
--output_dir $OUTPUT_DIR  \
--per_device_eval_batch_size=5 \
--per_device_train_batch_size=5 \
--gradient_accumulation_steps 2 \
--overwrite_output \
--cache_dir $CACHE_DIR \
--overwrite_cache \
--log_level info \
--logging_strategy epoch \
--save_strategy epoch \
--eval_strategy  epoch \
--seed $SEED \
--lr_scheduler_type $LR_SCHEDULER_TYPE \