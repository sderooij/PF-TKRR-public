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


class SeizureClassifier:
    def __init__(
        self,
        classifier: str,
        hyperparams: dict,
        cv_type: str,
        scaler=None,
        cv_obj=None,
        scoring="roc_auc",
        grid_search="full",
        n_jobs=-1,
        verbose=0,
    ):
        self.classifier = classifier
        self.hyperparams = hyperparams
        self.cv_type = cv_type
        self.cv_obj = cv_obj
        if self.cv_obj is None:
            self.cv_obj = LeaveOneGroupOut()
        self.scoring = scoring
        self.scaler = scaler
        if self.scaler is None:
            self.scaler = MinMaxScaler()
        self.grid_search = grid_search
        self.estimators = None
        self.results = None
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, features, labels, groups):
        results, estimators = classify(
            self.classifier,
            features,
            labels,
            groups,
            self.hyperparams,
            self.cv_obj,
            self.scaler,
            scoring=self.scoring,
            search=self.grid_search,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
        )
        self.estimators = estimators
        self.results = results
        return self

    def predict(self, features, groups, labels=False, concat=True):
        return predict_groups(
            self.estimators, features, groups, labels=labels, concat=concat
        )

    def score(self, predictions, labels, groups, scorers=None):
        scores = score_from_predictions(predictions, labels, groups, scorers)
        return scores

    def predict_and_score(self, features, labels, groups):
        predictions = self.predict(features, groups)
        scores = self.score(predictions, labels, groups)
        return predictions, scores

    def save(self, file):
        with open(file, "wb") as f:
            pickle.dump(self, f)
        return


def classify(
    clf,
    features,
    labels,
    groups,
    hyperparams: dict,
    cv,
    scaler,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1,
    search="full",
):
    """
    Patient specific cross validation using leave-one-seizure-out cross validation.

    Args:
        clf (sklearn classifier): classifier to use
        features (np.ndarray): features to use
        labels (np.ndarray): labels [-1,1]
        groups (np.ndarray): seizure groups
        hyperparams (dict): hyperparameters for grid search, e.g. {"C": [1,10], "gamma": ["scale"], "kernel": ["rbf"]}
        scoring (str): scoring metric, e.g. "roc_auc"
        cv (sklearn cross validation object): cross validation object, e.g. LeaveOneGroupOut()
        scaler (sklearn scaler): scaler to use, e.g. StandardScaler() or MinMaxScaler()
        n_jobs (int): default -1, number of jobs to run in parallel
        verbose (int): default 1, verbosity level, 0 = no messages, 1 = some messages, 2 = all messages
        search (str): default 'full', search strategy for grid search, 'full' = full grid search, 'random' = random search

    Returns:
        dict: cross validation results
        sklearn classifier: classifiers for each group / fold
    """
    pipe = Pipeline(steps=[("scaler", scaler), ("clf", clf)])
    if len(hyperparams) == 0:
        grid_search = pipe
    else:
        hyperparams = {f"clf__{key}": hyperparams[key] for key in hyperparams.keys()}
        if search == "full":
            grid_search = GridSearchCV(
                pipe, hyperparams, cv=5, scoring=scoring, n_jobs=n_jobs, verbose=verbose
            )
        elif search == "random":
            grid_search = RandomizedSearchCV(
                pipe, hyperparams, cv=5, scoring=scoring, n_jobs=n_jobs, verbose=verbose
            )
        elif search == "none":
            hyperparams = {key: hyperparams[key][0] for key in hyperparams.keys()}
            grid_search = pipe.set_params(**hyperparams)

    val_dict = cross_validate(
        grid_search,
        features,
        labels,
        groups=groups,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        return_estimator=True,
    )

    trained_clf = val_dict["estimator"]
    if search != "none":
        for estimator in trained_clf:
            print(estimator.best_params_)
    results = {key: val_dict[key] for key in val_dict.keys() if "test" in key}
    return results, trained_clf


def _call_decision_function(clf, features):
    return clf.decision_function(features)


def predict_groups(estimators, features, groups, labels=None, concat=True, cv_obj=LeaveOneGroupOut(), index=None):
    """
    Predict labels for each group using the estimators.

    Args:
        estimators (list): list of estimators, sklearn classifiers
        features (np.ndarray): features set, shape (n_samples, n_features)
        groups (np.ndarray): group ids, shape (n_samples,)
        labels (bool): true labels corresponding to the features
        concat (bool): default True, if True concatenate predictions to one array of shape (n_samples,), else return
        list of predictions for each group.

    Returns:
        np.ndarray: predictions (n_samples,) or (n_groups,)
    """

    num_groups = np.unique(groups).shape[0]
    num_feats = features.shape[0]
    # check if length correct
    assert len(estimators) == num_groups
    assert len(groups) == features.shape[0]
    #
    # group_features = []
    # group_idx = []
    # for i, group_id in enumerate(np.unique(groups)):  # separate features by group
    #     group_idx.append(np.where(groups == group_id))
    #     group_features.append(features[group_idx[i]])
    predictions = {}
    predictions['group_id'] = []
    predictions['prediction'] = []
    predictions['estimator'] = []
    predictions['label'] = []
    predictions['index'] = []
    for i, (train_idx, test_idx) in enumerate(cv_obj.split(X=features, y=labels, groups=groups)):
        predictions['group_id'].append(groups[test_idx])
        predictions['prediction'].append(estimators[i].decision_function(features[test_idx,:]))
        predictions['estimator'].append(i)
        predictions['label'].append(labels[test_idx])
        if index is None:
            predictions['index'].append(test_idx)
        else:
            predictions['index'].append(index[test_idx])

    # del features  # free memory
    # # predict decision_function for each group
    # with mp.Pool() as pool:
    #     predictions = pool.starmap(
    #         _call_decision_function, zip(estimators, group_features)
    #     )

    # reassemble predictions
    # if concat:
    #     new_pred = np.zeros((num_feats,))
    #     for i, group in enumerate(group_idx):
    #         new_pred[group] = predictions[i]
    #     predictions = new_pred
    #     del new_pred
    #
    # if labels:
    #     predictions = np.sign(predictions)
    return predictions


def score_from_predictions(predictions, labels, groups, scorers=None):
    """
    Calculate the scores for each group using the predictions.

    Args:
        predictions (list): predictions for each group, shape (n_groups,)
        labels (np.ndarray): labels [-1,1], true labels, shape (n_samples,)
        groups (np.ndarray): group ids, shape (n_samples,)
        scorers (list): list of scoring functions, e.g. [roc_auc_score, accuracy_score]

    Returns:
        dict: scores for each group
    """
    if scorers is None:
        scorers = [
            "accuracy",
            "f1",
            "roc_auc",
            "matthews_corrcoef",
            "precision",
            "recall",
        ]

    df_scores = pd.DataFrame(columns=scorers, index=np.unique(groups))
    num_groups = np.unique(groups).shape[0]
    assert len(predictions) == num_groups
    assert len(groups) == labels.shape[0]
    group_labels = []
    for group_id in np.unique(groups):
        group_idx = np.where(groups == group_id)
        group_labels.append(labels[group_idx])
        group_predictions = predictions[group_idx]
        for scorer in scorers:
            if scorer == "roc_auc":
                score_func = get_scorer(scorer)
                df_scores.loc[group_id, scorer] = score_func(
                    group_labels, group_predictions
                )
            else:
                score_func = get_scorer(scorer)
                df_scores.loc[group_id, scorer] = score_func(
                    group_labels, np.sign(group_predictions)
                )

    return df_scores


def plot_roc_curve_all_clf(predictions, labels, *, plot_chance=True):
    """
    Plot the roc curve for each group using the predictions.

    predictions (dict): predictions for each classifier, shape (n_clf,), where each entry is a ndarray of shape (n_samples,)
    labels (np.ndarray): labels [-1,1], true labels, shape (n_samples,)
    plot_chance (bool): default True, plot chance line

    Returns:
        matplotlib.axes.Axes: roc curve plot
    """
    ax = plt.gca()
    if isinstance(predictions, dict):
        for clf_id in predictions.keys():
            clf_disp = RocCurveDisplay.from_predictions(
                labels, predictions[clf_id], name=clf_id, ax=ax, plot_chance=plot_chance
            )
            clf_disp.plot()
    else:
        raise NotImplementedError

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    return ax


def remove_overlap_predictions(
    predictions: np.ndarray, feat_df: pd.DataFrame, group_df: pd.DataFrame
):
    """
    Remove overlapped features from the predictions
    Args:
        predictions (np.ndarray): predictions shape (n_samples,)
        feat_df (pd.DataFrame): feature dataframe with columns "index", "start_time", "stop_time"
        group_df (pd.DataFrame): group dataframe with columns "index", "group_id"

    Returns:
        np.ndarray: predictions without overlap, shape < (n_samples,)
    """
    # TODO: implement this
    raise NotImplementedError


def get_features(feature_file, group_file, cv_type="PS", *, patient_id=None):
    """Get the features and groups for the given patient. Or all patients if patient_id cv_type is 'PI'.

    Args:
        feature_file (str): location of parquet file with features
        group_file (str): location of parquet file with groups
        cv_type (str, optional): Cross-validation type, PS=patient-specific, PI=patient-independent. Defaults to 'PS'.
        patient_id (int, optional): ID of the patient (required for PS). Defaults to None.

    Returns:
        tuple: (DataFrame with features, DataFrame with groups)
    """

    if (cv_type == "PS" or cv_type == "PF") and patient_id is not None:
        feat_df = pd.read_parquet(
            feature_file, filters=[("Patient", "=", patient_id)]
        ).sort_values("index")
        group_df = pd.read_parquet(
            group_file, filters=[("Patient", "=", patient_id)]
        ).sort_values("index")

    elif cv_type == "PI":
        feat_df = pd.read_parquet(feature_file).sort_values("index")
        group_df = pd.read_parquet(group_file).sort_values("index")

    return feat_df, group_df


def extract_feature_group_labels(
    feat_df, group_df, *, delim_feat_chan="|", cv_type="PS"
):
    """
    Extract the feature and group labels from the feature and group dataframe.

    Args:
        feat_df (pd.DataFrame): feature dataframe with columns "index", "start_time", "stop_time"
        group_df (pd.DataFrame): group dataframe with columns "index", "group_id"
        delim_feat_chan (str): default '|', delimiter between feature and channel

    Returns:
        np.ndarray: features, shape (n_samples, n_features)
        np.ndarray: labels, shape (n_samples,)
        np.ndarray: groups, shape (n_samples,)
    """
    feat_cols = [col for col in feat_df.columns if delim_feat_chan in col]
    feats = feat_df.loc[:, feat_cols].to_numpy()
    labels = feat_df.loc[:, "annotation"].to_numpy()
    if cv_type == "PS" or cv_type == "PF":
        groups = group_df.loc[:, "group_id"].to_numpy()
    elif cv_type == "PI":
        groups = group_df.loc[:, "Patient"].to_numpy()
    return feats, labels, groups


def save_predictions(predictions, feature_df, *, save_file, delim_feat_chan="|", cv_type='PI'):
    feature_df.drop(
        columns=[col for col in feature_df.columns if delim_feat_chan in col],
        inplace=True,
    )
    if cv_type == 'PI' or cv_type == 'PS':
                # feature_df = feature_df.loc[:, ["index", "start_time", "stop_time", "annotation", "Patient"]]

        for i in range(len(predictions["group_id"])):
            feature_df.loc[feature_df['index'].isin(predictions["index"][i]), "prediction"] = predictions[
                "prediction"][i]
            feature_df.loc[feature_df['index'].isin(predictions["index"][i]), "estimator"] = predictions["estimator"][i]
            feature_df.loc[feature_df['index'].isin(predictions["index"][i]), "label"] = predictions["label"][i]

        # verify annotation and label the same
        assert np.all(feature_df["annotation"] == feature_df["label"])
        feature_df.to_parquet(save_file + ".parquet", index=True)
    elif cv_type == 'PF':   # leave P groups out

        dfs = []
        for i in range(len(predictions["group_id"])):
            temp_df = pd.DataFrame(columns=["index", "prediction", "estimator", "label"])
            temp_df.loc[:, "index"] = predictions['index'][i]
            temp_df.loc[:, 'prediction'] = predictions['prediction'][i]
            temp_df.loc[:, 'estimator'] = predictions['estimator'][i]
            temp_df.loc[:, 'label'] = predictions['label'][i]
            dfs.append(temp_df)
        new_df = pd.concat(dfs)
        new_df = new_df.merge(feature_df, on='index')
        new_df.to_parquet(save_file + ".parquet", index=False)
    return


def save_best_clf(
    estimators,
    folder,
    *,
    patient_id=None,
    estimator_names=None,
    classifier="svm",
    search="full",
):
    if not os.path.isdir(folder):
        os.makedirs(folder)

    elif estimator_names is None:
        best_clf = []
        for i, estimator in enumerate(estimators):
            if search != "none":
                best_clf.append(estimator.best_estimator_)
            else:
                best_clf.append(estimator)
    else:
        best_clf = {}
        for i, estimator in enumerate(estimators):
            if search != "none":
                best_clf[estimator_names[i]] = estimator.best_estimator_
            else:
                best_clf[estimator_names[i]] = estimator

    if patient_id is not None:
        with open(folder + f"{classifier}_{patient_id}.pickle", "wb") as f:
            pickle.dump(best_clf, f)
    else:
        with open(folder + f"{classifier}.pickle", "wb") as f:
            pickle.dump(best_clf, f)

    return


def main(
    feat_file,
    group_file,
    clf,
    hyperparam,
    scoring="roc_auc",
    scaler=MinMaxScaler(),
    cv_type="PS",
    cv_obj=LeaveOneGroupOut(),
    save_file=None,
    *,
    patient_id=None,
    delim_feat_chan="|",
    grid_search="full",
    model_folder=None,
    n_jobs=-1,
    classifier="cpkrr",
):
    feat_df, group_df = get_features(
        feat_file, group_file, cv_type=cv_type, patient_id=patient_id
    )
    features, labels, groups = extract_feature_group_labels(
        feat_df, group_df, cv_type=cv_type, delim_feat_chan=delim_feat_chan
    )
    if cv_type == "PF":
        n_groups = np.unique(groups).shape[0]
        cv_obj = LeavePGroupsOut(n_groups=n_groups - 1)  # leave one group in

    results, estimators = classify(
        clf,
        features,
        labels,
        groups,
        hyperparam,
        cv_obj,
        scaler,
        scoring=scoring,
        search=grid_search,
        verbose=0,
        n_jobs=-1,
    )

    if model_folder is not None:
        estimator_names = [f"{group}" for group in np.unique(groups)]
        save_best_clf(
            estimators,
            model_folder,
            patient_id=patient_id,
            classifier=classifier,
            estimator_names=estimator_names,
            search=grid_search,
        )
    predictions = predict_groups(
        estimators, features, groups, labels=labels, concat=True, cv_obj=cv_obj, index=feat_df['index'].to_numpy()
    )
    if save_file is not None:
        save_predictions(
            predictions, feat_df, save_file=save_file, delim_feat_chan=delim_feat_chan, cv_type=cv_type,
        )

    return predictions, labels, groups, results


if __name__ == "__main__":
    DELIM_FEAT_CHAN = "|"
    PATIENTS = [6514, 258, 5943, 5479, 1543, 6811]
    SCORING = "average_precision"

    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--cv_type",
            type=str,
            default="PS",
            help="PS=patient-specific, PI=patient-independent",
        )
        parser.add_argument("--feat_file", type=str)
        parser.add_argument("--group_file", type=str)
        parser.add_argument("--save_file", type=str)
        parser.add_argument("--hyper_param", type=str)
        parser.add_argument(
            "--grid_search", type=str, default="full", help="full or random"
        )
        parser.add_argument(
            "--classifier", type=str, default="svm", help="svm or cpkrr"
        )
        parser.add_argument("--model_folder", type=str, help="folder to save models")
        CV_TYPE = parser.parse_args().cv_type
        FEAT_FILE = parser.parse_args().feat_file
        GROUP_FILE = parser.parse_args().group_file
        SAVE_FILE = parser.parse_args().save_file
        HYPER_PARAM = parser.parse_args().hyper_param
        GRID_SEARCH = parser.parse_args().grid_search
        CLASSIFIER = parser.parse_args().classifier
        MODEL_FOLDER = parser.parse_args().model_folder

    else:  # no arguments given
        from seizure_data_processing.config import FEATURES_DIR

        FEAT_FILE = FEATURES_DIR + "features_ordered.parquet"
        GROUP_FILE = FEATURES_DIR + "val_groups.parquet"
        OUTPUT_DIR = "data/"
        CLASSIFIER = "svm"  # "svm" or "cpkrr"
        CV_TYPE = "PI"  # "PS" or "PI"
        GRID_SEARCH = "full"  # "True" or "False"

        # Set variables
        HYPER_PARAM = "data/hyper_parameters_" + CLASSIFIER + ".json"
        OUTPUT_FILE = OUTPUT_DIR + "results_" + CLASSIFIER + ".txt"
        MODEL_FOLDER = FEATURES_DIR + "models/" + CV_TYPE + "/"
        SAVE_FILE = FEATURES_DIR + CV_TYPE + "/results_" + CLASSIFIER

    if HYPER_PARAM == "None" or HYPER_PARAM == "none" or HYPER_PARAM is None:
        hyperparams = {}
    else:
        import json

        try:
            with open(HYPER_PARAM, "r") as f:
                hyperparams = json.loads(f.read())
        except:
            HYPER_PARAM = (
                "/users/selinederooij/Documents/src/seizure_data_processing/"
                + HYPER_PARAM
            )
            with open(HYPER_PARAM, "r") as f:
                hyperparams = json.loads(f.read())

    if CV_TYPE == "PS":
        print(hyperparams)
        for patient in PATIENTS:
            print(patient)
            if CLASSIFIER == "svm":
                clf = svm.SVC()
            elif CLASSIFIER == "cpkrr":
                clf = CPKRR()
            save_file = SAVE_FILE + f"_{patient}"
            predictions, labels, groups, results = main(
                FEAT_FILE,
                GROUP_FILE,
                clf,
                hyperparams,
                scoring=SCORING,
                cv_type=CV_TYPE,
                cv_obj=LeaveOneGroupOut(),
                save_file=save_file,
                patient_id=patient,
                delim_feat_chan=DELIM_FEAT_CHAN,
                grid_search=GRID_SEARCH,
                model_folder=MODEL_FOLDER,
                classifier=CLASSIFIER,
            )
            print(results)
    elif CV_TYPE == "PI":
        print(hyperparams)

        if CLASSIFIER == "svm":
            clf = svm.SVC()
        elif CLASSIFIER == "cpkrr":
            clf = CPKRR()
        save_file = SAVE_FILE
        predictions, labels, groups, results = main(
            FEAT_FILE,
            GROUP_FILE,
            clf,
            hyperparams,
            scoring=SCORING,
            cv_type=CV_TYPE,
            cv_obj=LeaveOneGroupOut(),
            save_file=save_file,
            delim_feat_chan=DELIM_FEAT_CHAN,
            grid_search=GRID_SEARCH,
            model_folder=MODEL_FOLDER,
            classifier=CLASSIFIER,
        )
        print(results)
