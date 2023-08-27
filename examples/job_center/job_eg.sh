#!/bin/bash
#SBATCH --account=ctb-liyue
#SBATCH --job-name=Ast1
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -e /home/austin29/projects/ctb-liyue/austin29/ATAC/main_method/DNA_BERT/examples/result/6/pretrain_$pretrain_ind/$data_preprocessing/Ast1/error
#SBATCH --output=/home/austin29/projects/ctb-liyue/austin29/ATAC/main_method/DNA_BERT/examples/result/6/pretrain_$pretrain_ind/$data_preprocessing/Ast1/%N-%j.out
#SBATCH --mail-user=onesmallstepforman13@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


module load python/3.8

cd /home/austin29/projects/ctb-liyue/austin29/ATAC/main_method/DNA_BERT
source .env/bin/activate
./examples/my_fine_tune_eg.sh Ast1
