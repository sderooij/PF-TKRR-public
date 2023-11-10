import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import pickle

sns.set_style("whitegrid")
matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     # "pgf.texsystem": "pdflatex",
#     # 'font.family': 'serif',
#     # 'text.usetex': True,
#     # 'pgf.rcfonts': False,
# })


def box_plot_scores(
    scores,
    hue="Classifier",
    save_file=None,
):

    plt.figure(figsize=(3.5, 3.5))
    ax = sns.boxplot(data=scores, y="Value", x="Metric", hue=hue, width=0.65, whis=5)   # don't remove outlier due to limited data
    ax.set_xlabel("")
    ax.set_ylabel("")
    hatches = ["//", "\\\\", "|"]
    # Loop over the bars
    for bars, hatch in zip(ax.containers, hatches):
        # Set a different hatch for each group of bars
        for bar in bars:
            bar.set_hatch(hatch)
    plt.xticks(rotation=0, fontsize=9)
    plt.yticks(np.arange(0, 1.1, step=0.1))
    # sns.ylabel("Average Precision Score")
    plt.tight_layout()
    ax.legend(loc="lower left", fancybox=False, ncol=2, fontsize=9, columnspacing=1.0)
    # plt.xtick_labels(fontsize)
    plt.show()
    if save_file is not None:
        plt.savefig(save_file, dpi=300)
    return


def bar_plot_scores(
    scores,
    hue="Classifier",
    save_file=None,
):
    plt.figure(figsize=(4.5, 2.5))
    ax = sns.barplot(data=scores, y="Value", x="Metric", hue=hue, errorbar=('ci', 95), palette="colorblind", capsize=.1, errwidth=1.)
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    plt.xticks(rotation=0, fontsize=11)
    plt.yticks(np.arange(0, 1.1, step=0.1), fontsize=11)
    # sns.ylabel("Average Precision Score")
    plt.tight_layout()
    ax.legend(loc="lower left", fancybox=False, ncol=2, fontsize=10, columnspacing=0.2, framealpha=1)
    # plt.xtick_labels(fontsize)
    if save_file is not None:
        plt.savefig(save_file, dpi=300) 
    else:
        plt.show()
    return



def num_floats_model(model_file, model_type="svm", pipeline=True):

    with open(model_file, "rb") as f:
        model = pickle.load(f)
    if model_type == "svm":
        n_floats = 0
        for pat, m in model.items():
            if pipeline:
                n_floats += (
                    m.named_steps["clf"].support_vectors_.size
                    + m.named_steps["clf"].dual_coef_.size
                )
            else:
                n_floats += m.support_vectors_.size + m.dual_coef_.size
    elif model_type == "cpkrr":
        n_floats = 0
        for pat, m in model.items():
            if pipeline:
                w = m.named_steps["clf"].weights_
            else:
                w = m.weights_
            n_floats += np.sum([a.size for a in w])
    else:
        raise ValueError("model_type must be either 'svm' or 'cpkrr'")

    n_floats /= len(model)
    return n_floats


if __name__ == "__main__":
    from src.config import FEATURES_DIR
    import itertools

    CV_TYPE = ["PI", "PF"]
    PATIENTS = [258, 1543, 5479, 5943, 6514, 6811]
    CLASSIFIER = ["cpkrr", "svm"]
    COUNT_MODEL_PARAMS = False
    cv_class = list(itertools.product(CV_TYPE, CLASSIFIER))

    # load scores
    scores = pd.DataFrame(columns=["Patient", "Metric", "Value", "Classifier"])
    for cv_type, classifier in cv_class:
        # if cv_type == "PF" and classifier == "svm":
            # continue
        score_df = pd.read_csv(f"{FEATURES_DIR}/{cv_type}/scores_{classifier}.csv")
        score_df.drop(columns=["estimator", 'accuracy'], inplace=True)
        # change names
        score_df.columns = ["Patient", "AUROC", "AUPRC", "F1", "Precision", "Sensitivity"]
        score_df = score_df.melt(id_vars=['Patient'], var_name='Metric', value_name='Value')
        if classifier == "cpkrr":
            score_df['Classifier'] = "T-KRR_{"+ cv_type + "}"
        elif classifier == "svm":
            if cv_type == "PF":
                cv_type = "PS"
            score_df['Classifier'] = "SVM_{"+ cv_type + "}"
        scores = pd.concat([scores, score_df])
    # box plot
        # if cv_type == 'PI':
        #     model_file = FEATURES_DIR + f"models/{cv_type}/{classifier}.pickle"
        # else:
        #     model_file = FEATURES_DIR + f"models/{cv_type}/{classifier}_1543.pickle"
        # n_floats = num_floats_model(model_file, model_type=classifier)
        # print(f"Number of floats {cv_type} {classifier}: {n_floats}")
    # box plot
    # bar_plot_scores(scores, hue="Classifier")
    bar_plot_scores(scores, hue="Classifier", save_file="bar_plot_scores.pgf")

    # print(scores.groupby(['Metric', 'Classifier']).mean())