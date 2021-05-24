export PATH=$PATH:$PWD
export CPDCTL_ENABLE_CODE_PACKAGE=1

cpdctl config context use cpd_prod

prod_space_id=$PROD_SPACE_ID

# Get ID of job for the existing code package
prod_job_id=$(cpdctl job list --space-id $prod_space_id --asset-ref-type code_package --raw-output --output json -j 'results[0].metadata.asset_id')

echo "Job ID: $prod_job_id"

cpdctl job run create --space-id $prod_space_id --job-id $prod_job_id --job-run '{}'