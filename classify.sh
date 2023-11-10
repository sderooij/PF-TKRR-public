#!/bin/bash

# This script runs the classification.py script and saves the output to a text file
# Global variables
FEATURES_DIR="/path/to/features/"
FEATURE_FILE="${FEATURES_DIR}features_ordered.parquet"
GROUP_FILE="${FEATURES_DIR}val_groups.parquet"
OUTPUT_DIR="data/"
CLASSIFIER="svm" # "svm" or "cpkrr"
CV_TYPE="PS" # "PS" or "PI"
GRID_SEARCH="full" # "full" or "random" or "none"

# Set variables
HYPERPARAM_FILE="data/hyper_parameters_${CLASSIFIER}.json"
OUTPUT_FILE="${OUTPUT_DIR}results_${CLASSIFIER}_${CV_TYPE}.txt"
MODEL_FOLDER="${FEATURES_DIR}models/${CV_TYPE}/"
SAVE_FILE="${FEATURES_DIR}${CV_TYPE}/results_${CLASSIFIER}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate seizure_env
start_time="$(date -u +%s)"
python src/scripts/classification.py --cv_type ${CV_TYPE} --classifier ${CLASSIFIER} --hyper_param ${HYPERPARAM_FILE} --feat_file ${FEATURE_FILE} --group_file ${GROUP_FILE} --grid_search ${GRID_SEARCH} --save_file=${SAVE_FILE} --model_folder=${MODEL_FOLDER} >> ${OUTPUT_FILE} 
end_time="$(date -u +%s)"

elapsed="$(($end_time-$start_time))"
echo "Total of $elapsed seconds elapsed for training"