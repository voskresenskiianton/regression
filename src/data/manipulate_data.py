import sys
import numpy as np
import pandas as pd

module_path = "../data/"
sys.path.append(module_path)
from load_data import get_config


def transform_data(df, clip=True, config_name="config"):
    """
    Prepare data for modelling.
    
    Parameters
    ----------
    df : DataFrame
        The final, canonical data sets for modeling.
    clip: bool
        
    Returns
    -------
    df : DataFrame
        Initial dataset.
    """
    config = get_config(name=config_name)

    if clip:
        for target in config["model_settings"]["target"]:
            df[target] = df[target].clip(0, 1)

    columns = ["RDEP"]
    for column in columns:
        df[column + "_log"] = np.log(df[column])
        df[column + "_sqrt"] = np.sqrt(df[column])

    return df


def get_predictors_and_target(df, feature_names, config_name="config"):
    """
    Split features and target into 2 dataframes.
    
    Parameters
    ----------
    df : DataFrame
        The final, canonical data sets for modeling.
    feature_names : str
        List containing имена колонок которые будут использованы как predictors.

    Returns
    -------
    X : DataFrame
        Features.
    y : Series
        Target.
    unique_groups : list
        Unique groups.
    """
    config = get_config(name=config_name)

    target = config["model_settings"]["target"]
    X, y = df[feature_names], df[target]

    group_name = config["model_settings"]["group"]
    unique_groups = df[group_name].unique()

    return X, y, unique_groups
