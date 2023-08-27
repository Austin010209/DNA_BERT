import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


def get_performance(cell_types, pretrain, preprocess):
    Y1 = {}
    for cell_type in cell_types:
        raw_result_dir = f"../result/6/pretrain_{pretrain}/{preprocess}/{cell_type}/test_results.tsv"
        if not os.path.exists(raw_result_dir):
            Y1[cell_type] = 0.5
            continue
        df = pd.read_csv(raw_result_dir, sep = "\t", index_col=False)
        max_auroc = np.max(np.array(df["auc"].tolist()))
        Y1[cell_type] = max_auroc
    return Y1


def main():
    print("Please check the data path again.")
    cell_types = os.listdir("../result/6/pretrain_True/None")
    cell_types = sorted(cell_types)
    # cell_types.remove('first_working')

    # pretrain=True, then it is "both", "seq", "label"
    # pretrain=False, then it is "None"
    pretrain_True_None = get_performance(cell_types, True, "None")
    # pretrain_False_None = get_performance(cell_types, False, "None")
    # pretrain_True_seq = get_performance(cell_types, True, "seq")
    # pretrain_True_label = get_performance(cell_types, True, "label")
    # pretrain_True_both = get_performance(cell_types, True, "both")


    DNABert_plot = [pretrain_True_None[i] for i in cell_types]

    f = open("test_results.tsv", "w")
    f.write("model\t" + "\t".join(cell_types) + "\n")
    f.write("pretrain_True_None_plot\t" + "\t".join([str(i) for i in DNABert_plot]) + "\n")
    f.close()
    df = pd.read_csv("test_results.tsv", sep = "\t")
    print(df)
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