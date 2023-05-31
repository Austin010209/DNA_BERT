import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


def get_performance(cell_types, pretrain, preprocess):
    Y1 = {}
    for cell_type in cell_types:
        raw_result_dir = f"../result/6/pretrain_{pretrain}/{preprocess}/{cell_type}/eval_results.tsv"
        if not os.path.exists(raw_result_dir):
            Y1[cell_type] = 0.5
            continue
        df = pd.read_csv(raw_result_dir, sep = "\t", index_col=False)
        max_auroc = np.max(np.array(df["auc"].tolist()))
        Y1[cell_type] = max_auroc
    return Y1

def main():
    cell_types = os.listdir("../result/6/pretrain_False/None")
    cell_types = sorted(cell_types)
    # cell_types.remove('first_working')

    # pretrain=True, then it is "both", "seq", "label"
    # pretrain=False, then it is "None"
    pretrain_False_None = get_performance(cell_types, False, "None")
    pretrain_True_seq = get_performance(cell_types, True, "seq")
    pretrain_True_label = get_performance(cell_types, True, "label")
    pretrain_True_both = get_performance(cell_types, True, "both")
    DNABert_result = {
        "Ast1": 0.829,  
        "Ast2": 0.903,
        "Ast3": 0.919,
        "Ast4": 0.91,
        "End1": 0.878,
        "End2": 0.86,
        "ExN1": 0.917,
        "ExN1_L23": 0.931,
        "ExN1_L24": 0.915,
        "ExN1_L46": 0.755,
        "ExN1_L56": 0.899,
        "ExN2": 0.902,
        "ExN2_L23": 0.922,
        "ExN2_L46": 0.897,
        "ExN2_L56": 0.9,
        "ExN3_L46": 0.909,
        "ExN3_L56": 0.901,
        "ExN4_L56": 0.896,
        "InN3": 0.868,
        "In_LAMP5": 0.91,
        "In_PV": 0.897,
        "In_SST": 0.902,
        "In_VIP": 0.899,
        "Mic1": 0.896,
        "Mic2": 0.861,
        "Oli1": 0.891,
        "Oli2": 0.888,
        "Oli3": 0.924,
        "Oli4": 0.937,
        "Oli5": 0.92,
        "Oli6": 0.877,
        "Oli7": 0.899,
        "OPC1": 0.878,
        "OPC2": 0.894,
        "OPC3": 0.902,
        "OPC4": 0.907
    }
    LSTM_result = {
        'Ast1': 0.7999924347094272, 
        'Ast2': 0.8871832085660173, 
        'Ast3': 0.9006020676033889, 
        'Ast4': 0.8880910677080835, 
        'End1': 0.5346748622094675, 
        'End2': 0.7133805170736863, 
        'ExN1': 0.7754483921992891, 
        'ExN1_L23': 0.9213229330952696, 
        'ExN1_L24': 0.8976610264972383, 
        'ExN1_L46': 0.7278079752383231, 
        'ExN1_L56': 0.8864789529828729, 
        'ExN2': 0.7522857720690842, 
        'ExN2_L23': 0.911061454693722, 
        'ExN2_L46': 0.8793631153766328, 
        'ExN2_L56': 0.8826660639068367, 
        'ExN3_L46': 0.7782151986863541, 
        'ExN3_L56': 0.8852672600189369, 
        'ExN4_L56': 0.87736482182532, 
        'InN3': 0.502789310222532, 
        'In_LAMP5': 0.8979553049541523, 
        'In_PV': 0.8814413580140426, 
        'In_SST': 0.8881476390702167, 
        'In_VIP': 0.8841277810439111, 
        'Mic1': 0.8798510433901156, 
        'Mic2': 0.8507085025313694, 
        'OPC1': 0.8564662081491642, 
        'OPC2': 0.5034043178492693, 
        'OPC3': 0.4926159313956202, 
        'OPC4': 0.5090144749127006, 
        'Oli1': 0.8736863597140039, 
        'Oli2': 0.8694746071829406, 
        'Oli3': 0.7916198866496027, 
        'Oli4': 0.9274380982165203, 
        'Oli5': 0.9075898234631772, 
        'Oli6': 0.8630401456724542, 
        'Oli7': 0.8842796010118736
    }


    DNABert_plot = [DNABert_result[i] for i in cell_types]
    LSTM_plot = [LSTM_result[i] for i in cell_types]
    pretrain_False_None_plot = [pretrain_False_None[i] for i in cell_types]
    pretrain_True_seq_plot = [pretrain_True_seq[i] for i in cell_types]
    pretrain_True_label_plot = [pretrain_True_label[i] for i in cell_types]
    pretrain_True_both_plot = [pretrain_True_both[i] for i in cell_types]
    print(cell_types)
    print(pretrain_False_None_plot)
    print(pretrain_True_seq_plot)
    exit()
    X = cell_types
    x_axis = np.arange(len(X))
    plt.figure(figsize=(9, 7))
    plt.bar(x_axis-0.4, DNABert_plot, width=0.2, label = 'DNABERT')
    plt.bar(x_axis-0.2, LSTM_plot, width=0.2, label = 'LSTM')
    plt.bar(x_axis, pretrain_False_None_plot, width=0.2, label = 'pretrain_False_None')
    plt.bar(x_axis+0.2, pretrain_True_seq_plot, width=0.2, label = 'pretrain_True_seq')
    
    plt.xticks(x_axis, X, rotation = 90)
    plt.ylabel("AUROC")
    plt.legend()
    plt.savefig("./auroc_bar_plot.png")
    plt.close()

    dif_LSTM = []
    dif_pretrain_False_None = []
    dif_pretrain_True_seq = []
    for i in cell_types:
        dif_LSTM.append(DNABert_result[i] - LSTM_result[i])
        dif_pretrain_False_None.append(DNABert_result[i] - pretrain_False_None[i])
        dif_pretrain_True_seq.append(DNABert_result[i] - pretrain_True_seq[i])
    plt.figure(figsize=(9, 7))
    plt.bar(x_axis-0.2, dif_LSTM, width=0.2, label = 'dif_LSTM')
    plt.bar(x_axis, dif_pretrain_False_None, width=0.2, label = 'dif_pretrain_False_None')
    plt.bar(x_axis+0.2, dif_pretrain_True_seq, width=0.2, label = 'dif_pretrain_True_seq')
    plt.xticks(x_axis, X, rotation = 90)
    plt.yticks(np.linspace(0, 0.5, 30))
    plt.ylabel("AUROC difference")
    plt.legend()
    plt.grid()
    plt.savefig("./dif.png")


if __name__ == "__main__":
    main()