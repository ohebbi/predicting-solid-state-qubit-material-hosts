import numpy as np
import logging
import sys
def clean_df(df):
    """Cleans dataframe by dropping missing values, replacing NaN's and infinities
    and selecting only columns containing numerical data.
    Args:
        df (pd.DataFrame): the dataframe to clean.
    Returns:
        pandas.DataFrame: the cleaned dataframe.
    """

    df = df.select_dtypes(exclude=['object'])
    df = df.replace([np.inf, -np.inf, np.nan], -1)
    df = df.dropna(axis=1, how="all")
    df = df.select_dtypes(include="number")
    return df

LOG = logging.getLogger("predicting-solid-state-qubit-candidates")
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
LOG.addHandler(handler)
