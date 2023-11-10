import numpy as np
import pandas as pd
import os
import sys
import argparse
import multiprocessing as mp

import matplotlib.pyplot as plt
import pickle

from sklearn import svm
from sklearn.model_selection import (
    LeaveOneGroupOut,
    LeavePGroupsOut,
    GridSearchCV,
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.metrics import (
    roc_auc_score,
    make_scorer,
    RocCurveDisplay,
    roc_curve,
    auc,
    get_scorer,
    average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tensorlibrary.learning.t_krr import CPKRR

# internal imports
import seizure_data_processing as sdp
import importlib

importlib.import_module("src/scripts/8_classification.py", package=None)


def main(
    feat_file,
    group_file,
    params,
    scoring="roc_auc",
    scaler=MinMaxScaler(),
    cv_type="PF",
    save_file=None,
    max_iter=50,
    w_init=None,
    *,
    patient_id=None,
    delim_feat_chan="|",
    classifier="cpkrr",
):
    feat_df, group_df = get_features(
        feat_file, group_file, cv_type=cv_type, patient_id=patient_id
    )
    features, labels, groups = extract_feature_group_labels(
        feat_df, group_df, cv_type=cv_type, delim_feat_chan=delim_feat_chan
    )
    n_groups = np.unique(groups).shape[0]
    cv_obj = LeavePGroupsOut(n_groups=n_groups - 1)  # leave one group in
    estimators = {}
    params["max_iter"] = 1
    results_df = pd.DataFrame(columns=["patient", "auc", "aucpr", "max_iter", 'fold'])
    total_predictions = np.zeros((features.shape[0], max_iter))
    for i, (train_idx, test_idx) in enumerate(cv_obj.split(features, labels, groups)):
        params["w_init"] = w_init
        print(f"Fold {i}")
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        estimators[f'{i}'] = []
        for iter in range(1, max_iter+1):
            params["max_iter"] = iter
            clf = CPKRR(**params)
            pipe = Pipeline(steps=[("scaler", scaler), ("clf", clf)])
            temp = pipe.fit(X_train, y_train)
            weights = temp.named_steps["clf"].weights_
            estimators[f"{i}"].append(pipe)
            params["w_init"] = weights
            # save results
            predictions = pipe.decision_function(X_test)
            aucroc = roc_auc_score(y_test, predictions)
            aucpr = average_precision_score(y_test, predictions)
            results_df = pd.concat([results_df, pd.DataFrame({"patient": [patient_id], "auc": [aucroc],
                                                              "aucpr": [aucpr], "max_iter": [iter], 'fold': [i]})])
            total_predictions[test_idx, iter-1] = predictions

    return results_df, total_predictions

    # results, estimators = classify(
    #     clf,
    #     features,
    #     labels,
    #     groups,
    #     hyperparam,
    #     cv_obj,
    #     scaler,
    #     scoring=scoring,
    #     search=grid_search,
    #     verbose=0,
    #     n_jobs=-1,
    # )

    # if model_folder is not None:
    #     estimator_names = [f"{group}" for group in np.unique(groups)]
    #     save_best_clf(
    #         estimators,
    #         model_folder,
    #         patient_id=patient_id,
    #         classifier=classifier,
    #         estimator_names=estimator_names,
    #         search=grid_search,
    #     )
    # predictions = predict_groups(
    #     estimators, features, groups, labels=False, concat=True
    # )
    # if save_file is not None:
    #     save_predictions(
    #         predictions, feat_df, save_file=save_file, delim_feat_chan=delim_feat_chan
    #     )
    #
    # return predictions, labels, estimators


if __name__ == "__main__":
    from seizure_data_processing.config import FEATURES_DIR
    MAX_ITER = 50
    CLASSIFIER = "cpkrr"
    MODE = 'random'
    model_file = FEATURES_DIR + f"models/PI/{CLASSIFIER}.pickle"
    with open(model_file, "rb") as f:
        models = pickle.load(f)
    # group_df = pd.read_parquet(FEATURES_DIR + "val_groups.parquet")
    feature_file = FEATURES_DIR + "features_ordered.parquet"
    group_file = FEATURES_DIR + "val_groups.parquet"
    results = pd.DataFrame(columns=["patient", "auc", "aucpr", "max_iter"])
    predictions = []
    tot_results = pd.DataFrame(columns=["patient", "auc", "aucpr", "max_iter"])
    for patient, model in models.items():
        print(patient)
        patient = int(patient)
        model.named_steps["clf"].max_iter = np.infty
        params = model.named_steps["clf"].get_params()
        params["num_sweeps"] = 10
        scaler = model.named_steps["scaler"]
        weights = model.named_steps["clf"].weights_
        if MODE == 'random':
            weights = None
        results, total_predictions = main(
            feature_file,
            group_file,
            params,
            scoring="roc_auc",
            scaler=scaler,
            cv_type="PF",
            save_file=None,
            w_init=weights,
            max_iter=MAX_ITER,
            patient_id=patient,
            delim_feat_chan="|",
            classifier=CLASSIFIER,
        )
        tot_results = pd.concat([tot_results, results])
        predictions.append(total_predictions)
    if MODE != 'random':
        tot_results.to_csv(FEATURES_DIR + f"PF/{CLASSIFIER}/results.csv")
    else:
        tot_results.to_csv(FEATURES_DIR + f"PF/{CLASSIFIER}/random_results.csv")