import json
import os
import boto3
# import numpy as np
# from langchain_community.document_loaders import UnstructuredMarkdownLoader
# from langchain_core import messages
import streamlit as st

class App():
    # TODO
    def __init__(self):
        st.title("Chat with document RAG")
        self.bedrock_client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_DEFAULT_REGION", None))
        self.llm = Claude3_Haiku(self.bedrock_client)
        self.chat()
        self.sidebar()

    def chat(self):
        # chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for x in st.session_state.messages:
            with st.chat_message(x['role']):
                st.write(x['content'][0]['text'])

        #  read user input
        if input_text := st.chat_input("Ask something here"):
            # print the user prompt
            with st.chat_message("user"):
                st.write(input_text)
            # add user prompt to chat history
            st.session_state.messages.append({
                'role':'user',
                'content':[{'text':input_text}]
            })

            # update the model parameters from the toggles in the sidebar
            self.llm.model_params = st.session_state.model_params

            # converse (send prompt along with chat history)
            output, input_tokens, output_tokens = self.llm.converse(st.session_state.messages, st.session_state.model_params)
            # append response to chat history
            st.session_state.messages.append(output)
            with st.chat_message("assistant"):
                st.write(output['content'][0]['text'])
            st.info(f"Input Tokens: {input_tokens} | Output Tokens: {output_tokens}")
            
            # before using bedrock_client.converse()
            # output, input_tokens, output_tokens = self.llm.generate_response(input_text)
            # # print the llm output
            # with st.chat_message("ai"):
            #     st.write(output)
            
    def sidebar(self):
        with st.sidebar:
            # upload file
            uploaded_file = st.file_uploader("Select file to chat with:")

            # model parameters
            st.subheader("Model Parameteres")
            self.configure_params_claude()

            # clear chat
            if st.button("Clear chat"):
                st.session_state.messages=[]
                for x in st.session_state.messages:
                    with st.chat_message(x['role']):
                        st.write(x['content'])

    def configure_params_claude(self):
        maxTokenCount_select = st.number_input("Configure model maxTokenCount:", min_value=0, max_value=4096, value=500, step=100)
        temperature_select = st.slider("Configure model temperature:", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        topP_select = st.slider("Configure model topP:", min_value=0.0, max_value=1.0, value=0.8, step=0.1)
        topK_select = st.slider("Configure model topK:", min_value=0, max_value=500, value=250, step=20)

        st.session_state.model_params = {
            "maxTokenCount":maxTokenCount_select,
            "temperature":temperature_select,
            "topP":topP_select,
            "topK":topK_select
        }

    def configure_params_titan(self):
        maxTokenCount_select= st.number_input("Configure model maxTokenCount:", min_value=0, max_value=3072, value=3072, step=100, key='a')
        temperature_select = st.slider("Configure model temperature:", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
        topP_select = st.slider("Configure model topP:", min_value=0.1, max_value=1.0, value=0.9, step=0.1)

        st.session_state.model_params = {
            "maxTokenCount":maxTokenCount_select,
            "stopSequences":[],
            "temperature":temperature_select,
            "topP":topP_select
        }

class ResponseError(Exception):
    def __init__(self, message):
        self.message = message

class Claude3_Haiku():
    def __init__(self,  bedrock_client, temperature=0.5, topP=0.8, topK=250, maxTokenCount=4096):
        self.bedrock = bedrock_client
        self.model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        self.model_params = {
            "maxTokenCount":maxTokenCount,
            "temperature":temperature,
            "topP":topP,
            "topK":topK
        }

    def generate_response(self, prompt):
        body = json.dumps({
            "prompt": f"\n\nHuman: {prompt} \n\nAssitant: ",
            "max_tokens_to_sample": self.model_params["maxTokenCount"],
            "temperature": self.model_params["temperature"],
            "top_p": self.model_params["topP"],
            "top_k": self.model_params["topK"]
        })

        response = self.bedrock.invoke_model(body=body, modelId=self.model_id)
        response_body =json.loads(response.get("body").read())

        finish_reason = response_body.get("error")
        if finish_reason is not None:
            raise ResponseError(f"Text generation error. Error is {finish_reason}")

        input_tokens = response_body['inputTextTokenCount']
        # collect the output tokens and the output
        output_tokens = 0   # initialize
        output = ""
        for x in response_body['results']:
            output_tokens += x['tokenCount']
            output = x['outputText']

        return output, input_tokens, output_tokens

    def converse(self, messages, model_params):
        # sys_prompt = [{
        #     "text": system_prompt
        # }]
        inference_params={
            "maxTokens":model_params["maxTokenCount"],
            "temperature": model_params["temperature"]
        }
        additional_params={
            "top_p": model_params["topP"],
            "top_k": model_params["topK"]
        }

        response = self.bedrock.converse(
            modelId=self.model_id,
            messages=messages,
            inferenceConfig=inference_params,
            additionalModelRequestFields=additional_params
        )

        output = response["output"]["message"]
        input_tokens=response["usage"]["inputTokens"]
        output_tokens=response["usage"]["outputTokens"]

        return output, input_tokens, output_tokens

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

        input_tokens = response_body['inputTextTokenCount']
        # collect the output tokens and the output
        output_tokens = 0   # initialize
        output = ""
        for x in response_body['results']:
            output_tokens += x['tokenCount']
            output = x['outputText']

        return output, input_tokens, output_tokens

class TitanEmbeddings():
    def __init__(self, bedrock_client, dimensions=1024, normalize=True, embeddingTypes=['float']):
        self.bedrock = bedrock_client
        self.model_id = "amazon.titan-embed-text-v2:0"
        self.model_params = {
            "dimensions":dimensions,
            "normalize":normalize,
            "embeddingTypes":embeddingTypes
        }

    def generate_embeddings(self, text):
        body =  {
            "inputText":text,
            "dimensions": self.model_params["dimensions"],
            "normalize": self.model_params["normalize"],
            "embeddingTypes": self.model_params["embeddingTypes"]
        }
        body = json.dumps(body)
        response = self.bedrock.invoke_model(body=body, modelId=self.model_id)
        response_body = json.loads(response.get('body').read())

        return response_body['embedding'], response_body['inputTextTokenCount']


# class DocumentLoader():
#     # TODO
#     def __init__(self):
#         pass
#         pass

def main():
    bedrock_client=boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_DEFAULT_REGION", None))
    streamlit_app = App()
    # embedding = TitanEmbeddings(bedrock_client)
    # inputString = "hello this is cool"
    # response, inputTokens = embedding.generate_embeddings(inputString)
    # print(response)
    # print(inputTokens)

if __name__ == "__main__":
    # TODO
    main()
