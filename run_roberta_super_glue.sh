export MODEL_NAME=roberta-large
export SEED=$1
export DATASET=$2
export LEARNING_RATE=2e-5
export EPOCH=6
export LR_SCHEDULER_TYPE=linear
export OUTPUT_DIR=$DATASET/super\_glue\_$MODEL_NAME\_lr.$LEARNING_RATE\_$LR_SCHEDULER_TYPE\_epoch.$EPOCH\_seed.$SEED\_log\_through\_time
export CACHE_DIR=temp\_dir

python3 run_roberta_super_glue.py \
--task_name $DATASET \
--seed $SEED \
--do_eval \
--do_train \
--model_name_or_path $MODEL_NAME \
--learning_rate $LEARNING_RATE \
--num_train_epochs $EPOCH \
--max_seq_length 512 \
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
--eval_strategy epoch \
--lr_scheduler_type $LR_SCHEDULER_TYPE \