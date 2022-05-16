import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


def mean_target_enc(df, n_bins, est=None, mean_target_PHIF=None):
    """
  Perform mean target encoding.
  """

    columns_to_transform = ["DEN"]
    if est is None:
        est = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
        est.fit(df[columns_to_transform])

    data = est.transform(df[columns_to_transform])
    data = pd.DataFrame(data, columns=columns_to_transform)
    data = data.add_suffix("_bins_" + str(n_bins))
    data.index = df.index

    df = df.join([data])

    column = "DEN_bins_" + str(n_bins)

    if mean_target_PHIF is None:
        mean_target_PHIF = df.groupby(column)["PHIF"].mean()

    df["DEN_PHIF_bins_" + str(n_bins) + "_mean_encoded"] = df[column].map(
        mean_target_PHIF
    )

    return df, est, mean_target_PHIF
