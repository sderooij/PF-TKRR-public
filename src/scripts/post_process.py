"""
    Remove overlapping segments and calculate the scores per patient.
"""

import numpy as np
# np.seterr(all='raise')
import pandas as pd
import warnings
warnings.simplefilter('error')

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, \
    precision_score, recall_score, fbeta_score, roc_curve, precision_recall_curve

from src.config import FEATURES_DIR


def load_predictions(classifier, cv_type, *, patients=[]):
    """
    Load the predictions from the given classifier and cv_type.
    """
    if cv_type == "PF" or cv_type == "PS":
        predictions = []
        for patient in patients:
            file = f"{FEATURES_DIR}/{cv_type}/results_{classifier}_{patient}.parquet"
            df = pd.read_parquet(file)
            predictions.append(df)
        predictions = pd.concat(predictions)
    elif cv_type == "PI":
        file = f"{FEATURES_DIR}/{cv_type}/results_{classifier}.parquet"
        predictions = pd.read_parquet(file)
    else:
        raise ValueError(f"cv_type {cv_type} not supported.")

    return predictions


def remove_overlap(df, seiz_overlap=2):
    index_keep = df.loc[df['annotation']==-1, 'index'].to_numpy()
    index_seiz = np.asarray(df.loc[df['annotation']==1, 'index'].unique(), dtype=int)
    index_seiz = np.sort(index_seiz)
    # keep every other index
    index_seiz = index_seiz[::seiz_overlap]

    # merge index keep and index seiz
    index = np.concatenate((index_keep, index_seiz))
    df = df.loc[df['index'].isin(index)]
    return df


def score_from_predictions(df, cv_type='PS'):

    results_df = pd.DataFrame(columns=["patient", "auc", "aucpr", "f1", "accuracy", "precision", "recall", "estimator"])
    for patient in df['Patient'].unique():
        pat_df = df.loc[df['Patient']==patient].copy()
        if cv_type =='PF':
            for estimator in pat_df['estimator'].unique():
                est_df = pat_df.loc[pat_df['estimator']==estimator].copy()
                bias = optimize_bias(est_df['prediction'].to_numpy(), est_df['annotation'].to_numpy(),
                                           metric='pr')
                prediction = est_df['prediction'].to_numpy()+bias
                predicted_label = np.sign(prediction) + (prediction == 0) # if bias==prediction, then label=0
                aucroc = roc_auc_score(est_df['annotation'], prediction)
                aucpr = average_precision_score(est_df['annotation'], prediction)
                f1 = f1_score(est_df['annotation'], predicted_label)
                acc = accuracy_score(est_df['annotation'], predicted_label)
                prec = precision_score(est_df['annotation'], predicted_label)
                rec = recall_score(est_df['annotation'], predicted_label)
                results_df = pd.concat([results_df, pd.DataFrame({"patient": [patient], "auc": [aucroc],
                                                                    "aucpr": [aucpr], "f1": [f1], "accuracy": [acc],
                                                                    "precision": [prec], "recall": [rec],
                                                                    "estimator": [estimator]})])

        else:
            bias = optimize_bias(pat_df['prediction'].to_numpy(), pat_df['annotation'].to_numpy(),
                                 metric='pr')
            prediction = pat_df['prediction'].to_numpy() + bias
            predicted_label = np.sign(prediction) + (prediction == 0)  # if bias==prediction, then label=0
            aucroc = roc_auc_score(pat_df['annotation'], prediction)
            aucpr = average_precision_score(pat_df['annotation'], prediction)
            f1 = f1_score(pat_df['annotation'], predicted_label)
            acc = accuracy_score(pat_df['annotation'], predicted_label)
            prec = precision_score(pat_df['annotation'], predicted_label)
            rec = recall_score(pat_df['annotation'], predicted_label)
            results_df = pd.concat([results_df, pd.DataFrame({"patient": [patient], "auc": [aucroc],
                                                              "aucpr": [aucpr], "f1": [f1], "accuracy": [acc],
                                                              "precision": [prec], "recall": [rec],
                                                              "estimator": [0]})])
    return results_df.groupby(['patient']).mean()
    # return results_df


def optimize_bias(predictions, labels, metric='roc', *, N=500):
    if metric == 'roc':
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        bias = thresholds[np.argmax(tpr - fpr)] # using Youden's J statistic
        bias = -bias

    elif metric == 'pr':
        precision, recall, thresholds = precision_recall_curve(labels, predictions, drop_intermediate=True)
        # get indices of zero and nan values
        zero_idx = np.where(precision <= 1e-5)[0]
        nan_idx = np.where(np.isnan(precision))[0]
        # remove zero and nan values
        precision = np.delete(precision, np.concatenate((zero_idx, nan_idx)))
        recall = np.delete(recall, np.concatenate((zero_idx, nan_idx)))
        thresholds = np.delete(thresholds, np.concatenate((zero_idx, nan_idx)))

        fscore = (2 * precision * recall) / (precision + recall)
        bias = thresholds[np.argmax(fscore)]

        assert np.isclose(f1_score(labels, np.sign(predictions-bias)+(predictions-bias == 0)),np.max(fscore))
        bias = -bias
    return bias



if __name__ == "__main__":
    CV_TYPE = ["PF"]
    PATIENTS = [258, 1543, 5479, 5943, 6514, 6811]
    CLASSIFIER = ["cpkrr"]
    SCORING = ["roc_auc", "average_precision", "f1", "accuracy", "precision", "recall"]

    # load predictions
    for cv_type in CV_TYPE:
        for classifier in CLASSIFIER:
            # if cv_type == "PF" and classifier == "svm":
            #     continue
            pred_df = load_predictions(classifier, cv_type, patients=PATIENTS)
            pred_df = remove_overlap(pred_df, seiz_overlap=2)
            scores = score_from_predictions(pred_df, cv_type=cv_type)
            scores.to_csv(f"{FEATURES_DIR}/{cv_type}/scores_{classifier}.csv")
            print(scores)



