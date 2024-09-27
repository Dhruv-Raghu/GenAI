import json
import os

import boto3
from langchain_aws.chat_models.bedrock import ChatBedrock

# first create the bedrock client
bedrock_client = boto3.client('bedrock-runtime', region_name=os.environ.get("AWS_DEFAULT_REGION", None))

# define the model id
modelId = "amazon.titan-text-premier-v1:0"

# define the model kwargs
# (setting them to the default values for now but we can change them later)
# Bedrock documentation to find inference params: https://tinyurl.com/2ya8fapf
kwargs = {"temperature": 0.7,
          "topP": 0.9,
          "maxTokenCount": 999}

# creating the model
chat = ChatBedrock(model_id=modelId, model_kwargs=kwargs)

# create the prompt structure
input_text = input("Type the phrase you would like to translate: \n")
language = input("What language would you like to translate this to? \n")

text = "'{input_text}'".format(input_text = input_text)
prompt = """User: Translate {text} to {language}
Translation: """.format(text=text, language=language)

# response = chat.invoke(prompt)
response = chat.stream(prompt)
# print(response.content)
for x in response:
    print(x, end="", flush=True)
