#!/bin/bash

cell_types=(Ast1 Ast2 Ast3 Ast4 End1 End2 ExN1 ExN1_L23 ExN1_L24 ExN1_L46 ExN1_L56 ExN2 ExN2_L23 ExN2_L46 ExN2_L56 ExN3_L46 ExN3_L56 ExN4_L56 In_LAMP5 InN3 In_PV In_SST In_VIP Mic1 Mic2 Oli1 Oli2 Oli3 Oli4 Oli5 Oli6 Oli7 OPC1 OPC2 OPC3 OPC4)
data_preprocessings=(None) #   both label seq     None

for cell_type in "${cell_types[@]}"
do
    for data_preprocessing in "${data_preprocessings[@]}"
    do
        # if [ $data_preprocessing = "None" ]
        # then
        #     pretrain_ind=False
        # else
        #     pretrain_ind=True
        # fi
        pretrain_ind=True
        ./job_gen.sh $cell_type $pretrain_ind $data_preprocessing
    done
done