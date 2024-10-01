import json
import os
import boto3
import numpy as np
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core import messages
import streamlit as st

class App():
    # TODO
    def __init__(self):
        st.title("Chat with document RAG")
        self.bedrock_client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_DEFAULT_REGION", None))
        self.llm = TitanText(self.bedrock_client)
        self.chat()
        self.sidebar()

    def chat(self):
        # chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for x in st.session_state.messages:
            with st.chat_message(x['role']):
                st.write(x['content'])

        if input_text := st.chat_input("Ask something here"):
            with st.chat_message("user"):
                st.write(input_text)

            st.session_state.messages.append({'role':'user', 'content':input_text})

            self.llm.model_params = st.session_state.model_params

            response = self.llm.generate_response(input_text)
            input_tokens = response['inputTextTokenCount']

            output_tokens = 0
            output = ""
            for x in response['results']:
                output_tokens += x['tokenCount']
                output = x['outputText']

            with st.chat_message("ai"):
                st.write(output)

            st.info(f"Input Tokens: {input_tokens}")
            st.info(f"Output Tokens: {output_tokens}")

            st.session_state.messages.append({'role':'ai', 'content':output})

    def sidebar(self):
        with st.sidebar:
            # upload file
            uploaded_file = st.file_uploader("Select file to chat with:")

            # model parameters
            st.subheader("Model Parameteres")
            maxTokenCount_select= st.number_input("Configure model maxTokenCount:", min_value=0, max_value=3072, value=3072, step=100, key='a')
            temperature_select = st.slider("Configure model temperature:", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
            topP_select = st.slider("Configure model topP:", min_value=0.1, max_value=1.0, value=0.9, step=0.1)

            st.session_state.model_params = {
                "maxTokenCount":maxTokenCount_select,
                "stopSequences":[],
                "temperature":temperature_select,
                "topP":topP_select
            }

            # clear chat
            if st.button("Clear chat"):
                st.session_state.messages=[]
                for x in st.session_state.messages:
                    with st.chat_message(x['role']):
                        st.write(x['content'])

class ResponseError(Exception):
    def __init__(self, message):
        self.message = message

class TitanText():
    def __init__(self, bedrock_client, maxTokenCount=3072, temp=0.7, topP=0.9):
        self.bedrock = bedrock_client
        self.model_id = "amazon.titan-text-premier-v1:0"
        self.model_params = {
            "maxTokenCount":maxTokenCount,
            "stopSequences":[],
            "temperature":temp,
            "topP":topP
        }
    def generate_response(self, prompt):
        body = json.dumps({
            "inputText": prompt,
            "textGenerationConfig": self.model_params
        })
        response = self.bedrock.invoke_model(body=body, modelId=self.model_id)
        response_body = json.loads(response.get("body").read())

        finish_reason = response_body.get("error")
        if finish_reason is not None:
            raise ResponseError(f"Text generation error. Error is {finish_reason}")

        return response_body

class TitanEmbeddings():
    def __init__(self, bedrock_client, ):
        bedrock = bedrock_client
        model_id = "amazon.titan-embed-text-v2:0"


class DocumentLoader():
    # TODO
    def __init__(self):
        pass
        pass

def main():
    bedrock_client=boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_DEFAULT_REGION", None))
    streamlit_app = App()

if __name__ == "__main__":
    # TODO
    main()
