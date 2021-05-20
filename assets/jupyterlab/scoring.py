from sklearn.pipeline import Pipeline
import datetime as dt
import pandas as pd
import numpy as np
import os
import sys


def find_project_dir():
    script_dir = None
    if os.path.isdir("project_git_repo"):
        script_dir = os.path.realpath("project_git_repo/cpd35-clustering-demo")
    else:
        script_dir = os.getcwd()
        
    return script_dir


PROJECT_DIR = find_project_dir()
SCRIPT_DIR = os.path.join(PROJECT_DIR, "assets/jupyterlab")
DATA_DIR = os.path.join(PROJECT_DIR, "assets/data_asset")
sys.path.append(os.path.normpath(SCRIPT_DIR))
print(SCRIPT_DIR)
print(DATA_DIR)

# from special_score.cpu_calculation import super_complexs_score
# from special_score.gpu_calculation import good_neigbour
from special_score.dbscan import my_dbscan
from preprocess_data import one_hot_encoder

reference_df = pd.read_csv(os.path.join(DATA_DIR, "credit_risk_reference.csv"))
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
    