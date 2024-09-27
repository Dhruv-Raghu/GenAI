import json
import os
import boto3


bedrock_client = boto3.client("bedrock_runtime", region_name=os.environ.get("AWS_DEFAULT_REGION", None))
