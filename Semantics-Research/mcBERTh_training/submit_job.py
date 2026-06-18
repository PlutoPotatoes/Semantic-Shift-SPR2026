from google.cloud import aiplatform
from google.oauth2 import service_account
import json

credentials_path = 'nlp-research-sp26.json'
credentials = service_account.Credentials.from_service_account_file(credentials_path)
with open(credentials_path) as f:
    service_account_email = json.load(f)['client_email']


# Set the tensorboard instance name 
tensorboard_name = "mcberth-test" # mcberth-tensorboard
existing_tb = aiplatform.Tensorboard.list(filter='display_name="{tensorboard_name}"')
tb = existing_tb[0] if existing_tb else aiplatform.Tensorboard.create(display_name=tensorboard_name)
tensorboard_resource = tb.resource_name


aiplatform.init(
    credentials=credentials,
    project="nlp-research-sp26",
    location="us-central1",
    staging_bucket="gs://project3102-model-bucket",
)

job = aiplatform.CustomContainerTrainingJob(
    display_name="mcberth-pretrain-v1-test",
    container_uri="us-docker.pkg.dev/nlp-research-sp26/mcberth-training/mcberth-training:test_mcberth_6_11_final",
)

job.run(
    machine_type="a2-highgpu-1g",
    accelerator_type="NVIDIA_TESLA_A100",
    accelerator_count=1,
    replica_count=1,
    base_output_dir="gs://project3102-model-bucket/Training-Tests/McBERTh-Pretrain-test",
    tensorboard=tensorboard_resource,
    service_account=service_account_email,
)

print("Job submitted successfully!")