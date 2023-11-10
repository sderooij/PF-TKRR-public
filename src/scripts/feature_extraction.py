import numpy as np
import pandas as pd
import multiprocessing as mp
import os
from functools import partial

# internal imports
import seizure_data_processing as sdp
from seizure_data_processing.pre_processing.features import extract_features
from src.config import TUSZ_DIR, FEATURES_DIR

# global variables
DEBUG = False
# FEATURES_DIR = r"U:/Seizure Data/v1.5.2/Features/tle_patients/"
PATIENTS = [6514, 258, 5943, 5479, 1543, 6811]


def load_file_names(ann_file, dir, patients=None, dataset="TUSZ"):
    if dataset == "TUSZ":
        df = pd.read_csv(ann_file, header=0)
        if patients is not None:
            df = df[df["Patient"].isin(patients)]
        df["Filename"] = df["Filename"].apply(lambda x: dir + x + ".edf")
        return df["Filename"].tolist()
    else:
        raise NotImplementedError


def load_parameters(param_file):
    df = pd.read_csv(param_file, header=0)
    return df.to_dict("records")[0]


def main(file, params):
    channels = params["channels"].split(";")
    eeg = sdp.EEG(file, channels)
    eeg = eeg.resample(params["Fs"])
    features = extract_features(
        eeg,
        params,
        window_time=params["window_time"],
        seiz_overlap=params["seiz_overlap"],
        bckg_overlap=params["bckg_overlap"],
        min_amplitude=params["min_amplitude"],
        max_amplitude=params["max_amplitude"],
    )

    return features


if __name__ == "__main__":
    file_names = load_file_names(
        "../data/annotations_TLE_patients.csv",
        TUSZ_DIR,
        patients=PATIENTS,
        dataset="TUSZ",
    )
    params = load_parameters("../data/parameters.csv")
    if DEBUG:
        feats = []
        for file in file_names:
            feats.append(main(file, params))
    else:
        pool_object = mp.Pool(mp.cpu_count())
        main_partial = partial(main, params=params)
        feats = pool_object.map(main_partial, file_names)
        pool_object.close()
        pool_object.join()
        feats = pd.concat(feats)
        feats.loc[:, "filename"].replace(TUSZ_DIR, "", regex=True, inplace=True)
        feats.loc[:, "filename"].replace(".edf", "", regex=True, inplace=True)
        feats = feats.reset_index(drop=True)  # useful for making sets
        feats.to_parquet(FEATURES_DIR + "features.parquet", index=True)
