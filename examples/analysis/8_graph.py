import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def main():
    # pretrain=True, then it is "both", "seq", "label"
    # pretrain=False, then it is "None"
    pretrained = True
    preprocessing_process = "label"
    raw_result_dir = f"../result/6/pretrain_{pretrained}/{preprocessing_process}"
    tasks = os.listdir(raw_result_dir)
    # tasks.remove('first_working')
    for task in tasks:
        do_one_task(task, pretrained, preprocessing_process, raw_result_dir)


def do_one_task(task, pretrained, preprocessing_process, raw_result_dir1):
    raw_result_dir = f"{raw_result_dir1}/{task}/eval_results.tsv"
    if not os.path.exists(raw_result_dir):
        print(f"{pretrained}, {preprocessing_process}, {task} is not here.")
        return
    df = pd.read_csv(raw_result_dir, sep = "\t", index_col=False)
    # metrics = ["accuracy", "recall", "precision", "f1", "bce_loss_train", "bce_loss_valid", "auprc", "auroc"]
    metrics = ["acc", "auc", "auprc", "f1", "precision", "recall", "valid_loss", "train_loss"]
    eval_data = {}
    for i in metrics:
        eval_data[i] = np.array(df[i].tolist()[:-1])
        if len(eval_data[i]) == 0:
            print(f"{pretrained}, {preprocessing_process}, {task} has 0 entry.")
            return
        # if i == "bce_loss_train":
        #     eval_data[i] = [j/1000 for j in eval_data[i]]
    X = list(range(1, len(df.index))) # it should have been 1+len(df.index), but
    # because the last entry is invalid, we minus one.
    # figure_dir = f"../graph/8_figures/{task}"
    figure_dir = f"../graph/8_figures/pretrain_{pretrained}/{preprocessing_process}"
    os.makedirs(figure_dir, exist_ok=True)

    max_metrics = ["acc", "recall", "precision", "f1", "auprc", "auc", "mcc"]
    min_metrics = ["valid_loss", "train_loss"]


    early_stopping_epoch = np.argmin(eval_data["valid_loss"])
    fig, axes = plt.subplots(2, 4, sharex = True)
    fig.set_size_inches(17, 7)
    for idx, i in enumerate(eval_data):
        row_num = idx // 4
        col_num = idx % 4
        ax = axes[row_num][col_num]
        y = eval_data[i]
        ax.plot(X, y, label = i)
        opt_value = "{:1.3f}".format(y[early_stopping_epoch])

        if i in max_metrics:
            max_val = "{:1.3f}".format(max(y))
            ax.set_title(f"{i} - opt {opt_value} (max {max_val})")
        elif i in min_metrics:
            min_val = "{:1.3f}".format(min(y))
            ax.set_title(f"{i} - opt {opt_value} (min {min_val})")
        
        ax.plot([early_stopping_epoch+1, early_stopping_epoch+1], list(ax.get_ylim()), '--')
        ax.plot([1, 1+len(df.index)], [y[early_stopping_epoch], y[early_stopping_epoch]], '--')
        ax.grid()
    fig.supxlabel("*5000 step")
    fig.suptitle(f"pretrain_{pretrained} {preprocessing_process} {task}")
    fig.savefig(os.path.join(figure_dir, f"{task}.png"))
    fig.clf()
    plt.close()

if __name__ == "__main__":
    main()