import numpy as np
import pandas as pd


def order_features(features, patient_info):
    """
    Order features by T1 and T2 dependent on patient -> T1 = Left, T2 = Right

    Args:
        features (pd.DataFrame): features to order
        patient_info (pd.DataFrame): patient information containing patient and side

    Returns:
        pd.DataFrame: ordered features
    """

    features["index"] = features.index
    patients = features["Patient"].unique()
    patient_info = patient_info.loc[patient_info["patient"].isin(patients)]
    # order features by T1 and T2 dependent on patient -> T1 = Left, T2 = Right
    feat_cols = [col for col in features.columns if DELIM_FEAT_CHAN in col]
    feat_df = features.loc[:, feat_cols].copy()
    idx = feat_df.columns.str.split(DELIM_FEAT_CHAN, expand=True)
    feat_df.columns = idx
    num_chan = 2
    feature_names = feat_df.columns.levels[0].unique()
    new_multi_idx = pd.MultiIndex.from_product([feature_names, range(num_chan)])
    feat_df.columns = new_multi_idx

    right_patients = patient_info.loc[
        patient_info["side"] == "right", "patient"
    ].unique()

    for i, feat in enumerate(feature_names):
        feat_df.loc[features["Patient"].isin(right_patients), feat] = np.fliplr(
            feat_df.loc[features["Patient"].isin(right_patients), feat].values
        )
        # for i, row in patient_info.iterrows():
        #     patient = row["patient"]
        #     side = row["side"]
        #     if side == "right":
        #         feat_i = np.fliplr(feat_df.loc[features['Patient'] == patient, feat].to_numpy())
        #         feat_df.loc[features['Patient'] == patient, feat] = feat_i

    feat_df.columns = [
        DELIM_FEAT_CHAN.join([str(val) for val in col])
        for col in feat_df.columns.values
    ]
    features.drop(columns=feat_cols, inplace=True)
    features = pd.concat([features, feat_df], axis=1, join="inner")

    return features


if __name__ == "__main__":
    # from seizure_data_processing.config import TUSZ_DIR
    from src.config import FEATURES_DIR
    PATIENT_FILE = "../data/TLE_patients.csv"
    ANN_FILE = "../data/annotations_TLE_patients.csv"
    DELIM_FEAT_CHAN = "|"

    # load files
    features = pd.read_parquet(FEATURES_DIR + "features.parquet")
    # load annotations to add Patient column to features
    annotations = pd.read_csv(ANN_FILE)
    annotations = annotations.rename(columns={"Filename": "filename"})
    annotations = annotations.loc[:, ["filename", "Patient"]].drop_duplicates()
    # join patient column of annotations to features on filename
    features = pd.merge(features, annotations, on="filename", how="left")
    del annotations
    # features.drop(columns=["Filename"], inplace=True)
    patient_info = pd.read_csv(PATIENT_FILE)
    # order features
    features = order_features(features, patient_info)

    # save features
    features.to_parquet(FEATURES_DIR + "features_ordered.parquet")
