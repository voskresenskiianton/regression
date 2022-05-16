import pandas as pd
import numpy as np
import yaml


def get_config(name="config"):
    """
    Get config file.

    Parameters
    ----------
    name : str
        Name of the config file.

    Returns
    -------
    config : dict
        Config file
    """
    module_path = "../data/"
    path = module_path + name + ".yml"
    try:
        with open(path, "r", encoding="utf8") as file:
            config = yaml.safe_load(file)
    except:
        print(path)
        raise ValueError("Error reading the config file. Check the file.")

    return config


def get_data(train=True):
    """
    Get and clean data.
    
    Parameters
    ----------
    name : str
        Nafe of teh config file.
    train : bool, default=True
        True for train, and False for test data.
    
    Returns
    -------
    df : DataFrame
        The final, canonical data sets for modeling.
    """
    config = get_config()

    path = config["model_settings"]["path_to_data"]

    if train:
        file_name = config["model_settings"]["train_file_name"]
    else:
        file_name = config["model_settings"]["test_file_name"]

    print(file_name)

    df = pd.read_csv(path + file_name)

    initial_df_shape = df.shape
    print("Shape of the initial dataset: {}.".format(initial_df_shape))

    df = df.drop_duplicates()
    print("Shape after duplicates dropping: {}.".format(df.shape))

    df = df.replace(["-9999", -9999, "NULL"], np.nan)
    df = df.reset_index(drop=True)

    columns_to_ignore = [config["model_settings"]["group"]]
    for item in (x for x in df.columns if x not in columns_to_ignore):
        df[item] = df[item].astype(float)

    processed_df_shape = df.shape
    print("Shape of the processed dataset: {}.".format(processed_df_shape))

    df = df.reset_index(drop=True)

    return df
