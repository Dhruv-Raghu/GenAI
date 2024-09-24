import streamlit as st
import pandas as pd
import numpy as np

import json
import os
import boto3
from langchain_aws.chat_models.bedrock import ChatBedrock

# initialize bedrock client
bedrock_client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_DEFAULT_REGION", None))
# define the model id
model_id = "amazon.titan-text-premier-v1:0"
# define the model kwargs
kwargs = {"temperature":0.7,
          "topP":0.9,
          "maxTokenCount":500
}
# create the model
translator = ChatBedrock(model_id=model_id, model_kwargs=kwargs)
# create the prompt template
prompt = "You are a translator. \n\nHuman: Translate '{input_text}' to {language} \n\nTranslation in {language}: "

# Create the GUI Interface
st.title("Amazon Titan Text as a Translator")

# Use selectbox in sidebar to choose language to translate to
with st.sidebar:
    language = st.selectbox("Choose a language to translate text to", ["French", "Spanish", "Italian"])

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for x in st.session_state.messages:
    with st.chat_message(x["role"]):
        st.write(x["content"])

# Create a chat input to allow the user to enter the input text
if input_text := st.chat_input("Enter text to translate"):
    with st.chat_message("user"):
        st.write(input_text)
    # append the message to session_state.messages
    st.session_state.messages.append({"role":"user", "content":input_text})

    response = translator.invoke(prompt.format(input_text=input_text, language=language))
    with st.chat_message("ai"):
        st.write(response.content)
    # optoken = response.response_metadata.usage_metadata.output_tokens
    # st.write(f"Input Tokens: {iptoken}, Output Tokens: {optoken}")

    st.session_state.messages.append({"role":"ai", "content":response.content})
