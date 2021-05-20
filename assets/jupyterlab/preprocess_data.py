from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np


def one_hot_encoder(input_df):
    features_idx = np.s_[0:-1]
    first_record_idx = np.s_[0]

    string_fields = [type(fld) is str for fld in input_df.iloc[first_record_idx, features_idx]]
    return ColumnTransformer([("ohe", OneHotEncoder(), list(np.array(input_df.columns)[features_idx][string_fields]))])