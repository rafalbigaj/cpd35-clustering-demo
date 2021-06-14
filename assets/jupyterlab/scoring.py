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

from special_score.dbscan import my_dbscan
from special_score.spectral import spectral
from training import train, evaluate, clusterings

reference_df = pd.read_csv(os.path.join(DATA_DIR, "credit_risk_reference.csv"))
input_df = reference_df.drop(['Risk'], axis=1)

# Training models and select winniong one

results = []

for (clustering_name, clustering_op) in clusterings:
    print(clustering_name)
    model = train(input_df, clustering_name, clustering_op)
    result = evaluate(reference_df, clustering_op)
    print("---")
    results.append(result)

best_score_idx = np.argmax(r['v_measure'] for r in results)
print("The winner is: '{}' with V-measure: {}!".format(clusterings[best_score_idx][0], results[best_score_idx]['v_measure']))