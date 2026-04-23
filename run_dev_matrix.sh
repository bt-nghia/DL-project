export SEED1=$1
export SEED2=$2
export DATASET=$3

# export FILE1=../super_glue_datasets/$DATASET/$DATASET\_$SEED1\_predict_results_id.csv
export FILE1=$DATASET/super_glue_meta-llama/Llama-3.2-3B_lr.5e-5_linear_epoch.20_seed.$SEED1\_predict_eval/predict_results_$DATASET.txt
export FILE2=$DATASET/super_glue_meta-llama/Llama-3.2-3B_lr.5e-5_linear_epoch.20_seed.$SEED2\_predict_eval/predict_results_$DATASET.txt

export OUTFILE=$DATASET/LLAMA_S_$SEED1\_$SEED2.txt
export GFILE=../super_glue_datasets/$DATASET/$DATASET\_validation.csv
# export GFILE=../glue_datasets/$DATASET/dev.csv

python3 run_dev_matrix.py \
--predfile1 $FILE1 \
--predfile2 $FILE2 \
--o $OUTFILE \
--g $GFILE \
