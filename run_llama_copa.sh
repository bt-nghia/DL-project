export MODEL_NAME=meta-llama/Llama-3.2-3B

export SEED=$1
export DATASET=copa
export LEARNING_RATE=3e-5
export EPOCH=10
export LR_SCHEDULER_TYPE=linear
export OUTPUT_DIR=$DATASET/super_glue_$MODEL_NAME\_lr.$LEARNING_RATE\_$LR_SCHEDULER_TYPE\_epoch.$EPOCH\_seed.$SEED\_predict_eval
export CACHE_DIR=temp\_dir


python3 run_llama_copa.py \
--task_name $DATASET \
--model_name_or_path $MODEL_NAME  \
--do_train  \
--do_eval  \
--do_predict \
--learning_rate $LEARNING_RATE \
--num_train_epochs $EPOCH  \
--max_seq_length 256  \
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