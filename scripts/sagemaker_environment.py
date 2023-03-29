from sagemaker import get_execution_role
import boto3
import sagemaker
from sagemaker.s3 import S3Uploader
from sagemaker.huggingface.model import HuggingFaceModel


class SageMakerEnvironment:
    def __init__(self, **kwargs):
        self._role = get_execution_role()
        self._region = boto3.Session().region_name
        self._sagemaker_session = sagemaker.session.Session()
        self._bucket = self.sagemaker_session.default_bucket()
        self._prefix = kwargs.get("prefix", "'huggingface-multimodel-deploy'")
        self._sm_client = boto3.client("sagemaker")


    def __str__(self):
        return f"sagemaker role arn: {self._role} \
                sagemaker bucket: {self._bucket} \
                sagemaker session region: {self._region}"
    
    def __repr__(self):
        return self.__str__()
    
    def upload_model_to_s3(self, model_path):
        return S3Uploader.upload(model_path, f"s3://{self.bucket}/{self.prefix}")
    
    def deploy_model(self, model_path, model_name, instance_type, instance_count, upload_model=True, **kwargs):
        if upload_model:
            model_path = self.upload_model_to_s3(model_path)
        model = sagemaker.model.Model(
            model_data=model_path,
            role=self.role,
            transformers_version=kwargs.get("transformers_version", "4.6.1"),
            pytorch_version=kwargs.get("pytorch_version", "1.7.1"),
            py_version=kwargs.get("py_version", "py38"),
        )

        predictor = model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
        )
        
        return predictor


    @property
    def sm_client(self):
        return self._sm_client
    
    @sm_client.setter
    def sm_client(self, value):
        self._sm_client = value

    @property
    def sagemaker_session(self):
        return self._sagemaker_session
    
    @sagemaker_session.setter
    def sagemaker_session(self, value):
        self._sagemaker_session = value

    @property
    def bucket(self):
        return self._bucket
    
    @bucket.setter
    def bucket(self, value):
        self._bucket = value

    @property
    def prefix(self):
        return self._prefix
    
    @prefix.setter
    def prefix(self, value):
        self._prefix = value

    @property
    def region(self):
        return self._region
    
    @region.setter
    def region(self, value):
        self._region = value

    @property
    def role(self):
        return self._role
    
    @role.setter
    def role(self, value):
        self._role = value



if __name__ == "__main__":
    sm_env = SageMakerEnvironment()
    print(sm_env)
