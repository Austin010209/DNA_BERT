#!/bin/bash
#SBATCH --account=ctb-liyue
#SBATCH --job-name=single_test
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -e /home/austin29/projects/ctb-liyue/austin29/ATAC/main_method/DNA_BERT/examples/job_center/error
#SBATCH --output=/home/austin29/projects/ctb-liyue/austin29/ATAC/main_method/DNA_BERT/examples/job_center/%N-%j.out
#SBATCH --mail-user=onesmallstepforman13@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


module load python/3.8

cd /home/austin29/projects/ctb-liyue/austin29/ATAC/main_method/DNA_BERT
source .env/bin/activate
cd ./examples


cell_types=(ExN1_L56 ExN2 ExN2_L23 ExN2_L46 ExN2_L56 ExN3_L46 ExN3_L56 ExN4_L56 In_LAMP5 InN3 In_PV In_SST In_VIP Mic1 Mic2 Oli1 Oli2 Oli3 Oli4 Oli5 Oli6 Oli7 OPC1 OPC2 OPC3 OPC4)

for cell_type in "${cell_types[@]}"
do
    ./my_fine_tune_eval.sh ${cell_type} True None
done