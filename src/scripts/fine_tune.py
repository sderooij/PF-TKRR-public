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
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tensorlibrary.learning.t_krr import CPKRR

# internal imports
import seizure_data_processing as sdp
import importlib

importlib.import_module("src/scripts/5_classification.py", package=None) # import main


def patient_fine_tune(model, num_sweeps=1, max_iter=np.infty):

    if isinstance(model, Pipeline):
        old_clf = model.named_steps["clf"]
        scaler = model.named_steps["scaler"]
    else:
        old_clf = model
        scaler = MinMaxScaler()
    # get params of old model and set new params
    old_clf.max_iter = np.infty
    weights = old_clf.weights_
    params = old_clf.get_params()
    params["num_sweeps"] = num_sweeps
    params["max_iter"] = max_iter
    params["w_init"] = weights
    clf = CPKRR(**params)

    return clf, scaler


if __name__ == "__main__":
    from src.config import FEATURES_DIR

    CLASSIFIER = "cpkrr"
    MODE = None
    # NUM_SWEEPS = 2
    MAX_ITER = [25]
    N_SWEEPS = 1
    model_file = FEATURES_DIR + f"models/PI/{CLASSIFIER}.pickle"
    with open(model_file, "rb") as f:
        models = pickle.load(f)
    # group_df = pd.read_parquet(FEATURES_DIR + "val_groups.parquet")
    feature_file = FEATURES_DIR + "features_ordered.parquet"
    group_file = FEATURES_DIR + "val_groups.parquet"
    model_folder = FEATURES_DIR + "models/PF/"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    for max_iter in MAX_ITER:
        for patient, model in models.items():
            print(patient)
            patient = int(patient)
            if CLASSIFIER == "cpkrr" and MODE != "random":
                clf, scaler = patient_fine_tune(model, num_sweeps=N_SWEEPS, max_iter=max_iter)
            elif CLASSIFIER == "cpkrr" and MODE == "random":
                model.named_steps["clf"].max_iter = np.infty
                params = model.named_steps["clf"].get_params()
                params["num_sweeps"] = 10
                params['max_iter'] = max_iter
                clf = CPKRR(**params)
                scaler = model.named_steps["scaler"]
            elif CLASSIFIER == "svm":
                params = model.named_steps["clf"].get_params()
                params['gamma'] = 0.4
                clf = svm.SVC(**params)
                scaler = model.named_steps["scaler"]

            save_file = FEATURES_DIR + f"PF/results_{CLASSIFIER}_{patient}"

            predictions, labels, groups, results = main(
                feature_file,
                group_file,
                clf,
                {},
                scaler=scaler,
                patient_id=patient,
                cv_type="PF",
                scoring="roc_auc",
                grid_search="none",
                model_folder=model_folder,
                save_file=save_file,
                classifier=CLASSIFIER,
            )
            print(results)
        # features, groups = get_features(feature_file, group_file, cv_type="PS", patient=patient)
        # features, labels, groups = extract_feature_group_labels(features, groups, cv_type="PS", delim_feat_chan="|")
        # results, trained_clf = patient_fine_tune(model, features, labels, groups, num_sweeps=1)
        # estimator_names = [f"{group}" for group in np.unique(groups)]
        # save_best_clf(trained_clf, estimator_names, patient, model_file)
        # predictions = predict_groups(trained_clf, features, patient_id=patient, estimator_names=estimator_names)
