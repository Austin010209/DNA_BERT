#!/bin/bash
#SBATCH --account=ctb-liyue
#SBATCH --job-name=CELL_TYPE_PRETRAIN_IND_DATA_PREPROCESSING
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH -e /home/austin29/projects/ctb-liyue/austin29/ATAC/main_method/DNA_BERT/examples/result/6/pretrain_PRETRAIN_IND/DATA_PREPROCESSING/CELL_TYPE/error
#SBATCH --output=/home/austin29/projects/ctb-liyue/austin29/ATAC/main_method/DNA_BERT/examples/result/6/pretrain_PRETRAIN_IND/DATA_PREPROCESSING/CELL_TYPE/%N-%j.out
#SBATCH --mail-user=onesmallstepforman13@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


module load python/3.8

cd /home/austin29/projects/ctb-liyue/austin29/ATAC/main_method/DNA_BERT
source .env/bin/activate
./examples/my_fine_tune_eg_pretrain_None.sh CELL_TYPE PRETRAIN_IND DATA_PREPROCESSING
# my_fine_tune_no_pretrain 
# my_fine_tune_eg
# my_fine_tune_eg_pretrain_None.sh