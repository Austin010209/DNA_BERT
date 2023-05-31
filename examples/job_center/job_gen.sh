#!/bin/bash

cell_type=$1
pretrain_ind=$2
data_preprocessing=$3


# local_dir=./all_jobs
local_dir=./pretrain_None_jobs
file_name="$local_dir/${cell_type}_${pretrain_ind}_${data_preprocessing}.sh"

echo "$file_name"
sed -e "s/CELL_TYPE/$cell_type/g" -e "s/PRETRAIN_IND/$pretrain_ind/g" -e "s/DATA_PREPROCESSING/$data_preprocessing/g" target.sh > $file_name
chmod u+x $file_name