from sklearn.pipeline import Pipeline
import datetime as dt
import pandas as pd
import numpy as np
import os
import sys


def find_script_dir():
    script_dir = None
    try: script_dir = os.path.dirname(os.path.realpath(__file__))
    except NameError: script_dir = os.path.realpath("project_git_repo/cpd35-clustering-demo/assets/jupyterlab")
        
    return script_dir


SCRIPT_DIR = find_script_dir()
print(SCRIPT_DIR)
sys.path.append(os.path.normpath(SCRIPT_DIR))

# from special_score.cpu_calculation import super_complexs_score
# from special_score.gpu_calculation import good_neigbour
from special_score.dbscan import my_dbscan
from preprocess_data import one_hot_encoder

reference_df = pd.read_csv(os.path.join(SCRIPT_DIR, "data/credit_risk_reference.csv"))
input_df = reference_df.drop(['Risk'], axis=1)

transformer = one_hot_encoder(input_df)

# Training an unsupervised model
# Applying an unsupervised model for inference
dbscan_model = my_dbscan()

pipeline_linear = Pipeline([('transformer', transformer), ('dbscan', dbscan_model)])
model = pipeline_linear.fit(input_df)

# One score that is compute intensive and has to run on GPU
# buddy = good_neigbour(trades_prepared_filtered_df['account', 'date', 'volume'])

print(np.unique(dbscan_model.labels_))
