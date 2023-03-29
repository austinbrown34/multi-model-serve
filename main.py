import argparse
from config import SELECTED_MODELS
import os
import tarfile
from distutils.dir_util import copy_tree
from scripts.sagemaker_environment import SageMakerEnvironment


def main(args):
    # Create SageMaker environment
    sagemaker_env = SageMakerEnvironment()
    # Download models
    for model in args.models:
        SELECTED_MODELS[model]["download"](args.model_dir)
        # Add code folder to model dirs
        os.makedirs(os.path.join(args.model_dir, model, "code"), exist_ok=True)
        # Copy model code to model dir
        copy_tree(
            os.path.join("code", model),
            os.path.join(args.model_dir, model, "code")
        )
        # Tar model dir
        with tarfile.open(os.path.join(args.model_dir, model + ".tar.gz"), "w:gz") as tar:
            tar.add(os.path.join(args.model_dir, model), arcname=model)
    
    for model in args.models:
        # Upload model to S3
        sagemaker_env.upload_model(os.path.join(args.model_dir, model + ".tar.gz"), model)

    # Create Multi-Model Endpoint
    # create SageMaker Model
    image_uri = "763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04"
    multimodels_path = 's3://sagemaker-us-east-2-(account-id)/huggingface-multimodel-deploy/models/'
    deployment_name = "huggingface-multi-model"

    primary_container = {
        'Image': image_uri,
        'Mode': 'MultiModel',
        'ModelDataUrl': multimodels_path,
        'Environment': {
            'SAGEMAKER_PROGRAM': 'inference.py',
            'SAGEMAKER_REGION': sagemaker_env.region,
            'SAGEMAKER_SUBMIT_DIRECTORY': multimodels_path
        }
    }

    create_model_response = sagemaker_env.sm_client.create_model(
        ModelName = deployment_name,
        ExecutionRoleArn = sagemaker_env.role,
        PrimaryContainer = primary_container
    )

    print(create_model_response['ModelArn'])

    # Create SageMaker Endpoint Config
    endpoint_config_response = sagemaker_env.sm_client.create_endpoint_config(
        EndpointConfigName = deployment_name,
        ProductionVariants = [{
            'InstanceType': 'ml.m5.large',
            'InitialInstanceCount': 1,
            'InitialVariantWeight': 1,
            'ModelName': deployment_name,
            'VariantName': 'AllTraffic'
        }]
    )

    print("Endpoint Config Arn: " + endpoint_config_response['EndpointConfigArn'])

    # Create SageMaker Endpoint
    endpoint_params = {
        'EndpointName': deployment_name,
        'EndpointConfigName': deployment_name
    }

    endpoint_response = sagemaker_env.sm_client.create_endpoint(**endpoint_params)
    print("Endpoint Arn: " + endpoint_response['EndpointArn'])
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=SELECTED_MODELS.keys(),
        help='List of models include in endpoint'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='models',
        help='Directory to download models'
    )

    args = parser.parse_args()
    main(args)
