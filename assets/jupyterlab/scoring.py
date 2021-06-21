import pandas as pd
import os
import sys
import types
from botocore.client import Config
import ibm_boto3


def find_project_dir():
    if os.path.isdir("project_git_repo"):
        return os.path.realpath("project_git_repo/cpd35-clustering-demo")
    else:
        return os.getcwd()


PROJECT_DIR = find_project_dir()
SCRIPT_DIR = os.path.join(PROJECT_DIR, "assets/jupyterlab")
DATA_DIR = os.path.join(PROJECT_DIR, "assets/data_asset")
sys.path.append(os.path.normpath(SCRIPT_DIR))
print(SCRIPT_DIR)
print(DATA_DIR)

from training import train, evaluate, clusterings


cos_endpoint = 'https://s3.us.cloud-object-storage.appdomain.cloud'

client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='lxZVqSmOBU6ta1oOlvz2yaHzHV-qYh2zx02ZnCVmmywC',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url=cos_endpoint)

body = client.get_object(Bucket='orchestrationbafindemo-donotdelete-pr-6h6r6ovdlfuhih',Key='credit_risk_reference.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

reference_df = pd.read_csv(body)
input_df = reference_df.drop(['Risk'], axis=1)

with open(os.path.join(PROJECT_DIR, 'selected_algorithm')) as file:
    clustering_name = file.read().splitlines()

print("Using preselected clustering algorithm: {}".format(clustering_name))

clustering_op = clusterings[clustering_name]

model = train(input_df, clustering_name, clustering_op)
labels_pred = clustering_op.labels_
input_df['Risk'] = labels_pred

print(input_df)
