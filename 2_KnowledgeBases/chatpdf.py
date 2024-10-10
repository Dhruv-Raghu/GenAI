import os
import boto3
import streamlit as st
import pymupdf
from models import Claude3_Haiku
from chunker import SemanticChunker

class App():
    # TODO
    def __init__(self):
        st.title("Chat with document RAG")
        self.bedrock_client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_DEFAULT_REGION", None))
        self.llm = Claude3_Haiku(self.bedrock_client)
        # st.session_state.file = None
        self.sidebar()
        self.chat()

    def chat(self):
        # chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for x in st.session_state.messages:
            with st.chat_message(x['role']):
                st.write(x['content'][0]['text'])

        if st.session_state.file is not None:
            doc = pymupdf.open(stream=st.session_state.file.read(), filetype='pdf')

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
            uploaded_file = st.file_uploader("Select file to chat with:", type='pdf', key='file') #noqa

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

# class DocumentLoader():
#     # TODO
#     def __init__(self):
#         pass
#         pass

def main():
    # bedrock_client=boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_DEFAULT_REGION", None))
    streamlit_app = App()
    # embedding = TitanEmbeddings(bedrock_client)
    # inputString = "hello this is cool"
    # response, inputTokens = embedding.generate_embeddings(inputString)
    # print(response)
    # print(inputTokens)

if __name__ == "__main__":
    # TODO
    main()
