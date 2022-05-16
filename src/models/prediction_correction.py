import numpy as np
import pandas as pd
import sys

module_path = "../data/"
sys.path.append(module_path)
from load_data import get_config


def apply_heuristics(prediction, df_val, df, statistics):
    """
    Perform heuristics on prediction dataset
    """
    config = get_config()
    group = config["model_settings"]["group"]

    # PHIF
    # make the values lie in the range < 0 and > 95 pecentiles
    prediction["PHIF"] = prediction["PHIF"].clip(0, df["PHIF"].quantile(0.95))
    # make values which have `DEN` values > 3 equal to 0.02
    prediction["PHIF"] = np.where(
        (df_val["DEN"] > 3),
        df[df["DEN"] > 3][["PHIF"]].mean().values[0],
        prediction["PHIF"],
    )

    # SW
    well = 100
    min_index = df_val[
        df_val[group] == well
    ].index.min()  # get minimum index of well 100
    max_index = df_val[
        df_val[group] == well
    ].index.max()  # get maximum index of well 100
    # make `SW` in well 100 equal to 1
    prediction.loc[(df_val.index >= min_index) & (df_val.index <= max_index), "SW"] = 1
    # PERFORM post-prediction correction
    prediction["SW"] = prediction["SW"] + statistics["SW_error"].values[0]
    # make `SW` values which have `RDEP_log` values < -2 equal to 1
    prediction["SW"] = np.where((df_val["RDEP_log"] < -2), 1, prediction["SW"])

    # VSH
    # make `VSH` values which have `GR` values > max value in train dataset equal to 1
    prediction["VSH"] = np.where((df_val["GR"] > df["GR"].max()), 1, prediction["VSH"])

    return prediction
