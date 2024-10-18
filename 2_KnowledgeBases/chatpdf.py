import os

import boto3
import pymupdf
from document_loaders import PdfLoader
import streamlit as st
from chunker import SemanticChunker, LateChunker
from fmodels import Claude3_Haiku, TitanEmbeddings
from pymongo.cursor import OperationFailure
from vectordb import MongoDB

from transformers import AutoModel


class App():
    # TODO
    def __init__(self):
        st.title("Chat with document RAG")
        self.bedrock_client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_DEFAULT_REGION", None))
        self.llm = Claude3_Haiku(self.bedrock_client)
        self.mongodb = MongoDB(database_name="chatwpdf", collection_name="uploaded_docs")
        self.prompt = ''
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

        #  read user input
        if input_text := st.chat_input("Ask something here"):
            # print the user prompt
            with st.chat_message("user"):
                st.write(input_text)

            self.prompt = input_text

            # update the model parameters from the toggles in the sidebar
            self.llm.model_params = st.session_state.model_params

            # if pdf has been added
            if st.session_state.file is not None:
                # st.success('Document Uploaded!', icon="✅")

                # initialize the embedding model and embed the query
                model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
                query_embed = model.encode(input_text)
                # perform a vector search on the database and return the 3 most relevant chunks
                vector_search = self.mongodb.retrieve(
                    index_name='vector_index',
                    query_embedding=query_embed.tolist(),
                    embedding_field='embedding',
                    num_neighbors=100,
                    limit=3
                )
                # add the relevant context to the prompt
                context = []
                for results in vector_search:
                    context.append(results['text'])
                context_to_add = f"chunk 1:{context[0]}, chunk 2:{context[1]}, chunk 3:{context[2]}"
                self.prompt = f"Answer <{input_text}> using relevant context from <Context from pdf->{context_to_add}>"

            # add user prompt to chat history
            st.session_state.messages.append({
                'role':'user',
                'content':[{'text':self.prompt}]
            })
            # converse (send prompt along with chat history)
            output, input_tokens, output_tokens = self.llm.converse(st.session_state.messages, st.session_state.model_params)
            # remove the context from the message history after the model output has been received (save input token costs down the line)
            st.session_state.messages[-1]['content'][0]['text'] = input_text
            # append response to chat history
            st.session_state.messages.append(output)
            with st.chat_message("assistant"):
                st.write(output['content'][0]['text'])
            st.info(f"Input Tokens: {input_tokens} | Output Tokens: {output_tokens}")


    def sidebar(self):
        with st.sidebar:
            # upload file
            # uploaded_file = st.file_uploader("Select file to chat with:", type='pdf', key='file', on_change=self.process_document) #noqa
            uploaded_file = st.file_uploader("Select file to chat with:", type='pdf', key='file', on_change=self.process_document) #noqa

            # model parameters
            st.subheader("Model Parameteres")
            self.configure_params_claude()

            # options to clear chat history and pdf in the database
            with st.expander("Clear"):
                # clear chat
                if st.button("Clear chat history"):
                    self.clear('chat')
                if st.button("Clear Database"):
                    self.clear('collection')

    def configure_params_claude(self):
        # configurable params for the claude model
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

    def process_document(self):
        # check if file has been uploaded
        if st.session_state.file is not None:
            # initialize the pymupdf doc
            with st.spinner('Opening Document...'):
                doc = pymupdf.open(stream=st.session_state.file.read(), filetype='pdf')

            # perform late chunking on the document
            with st.spinner('Chunking Document'):
                late_chunker = LateChunker()
                chunks, chunk_embeddings = late_chunker.get_chunk_embeddings(doc)

            # upload the chunks and their embeddings to mongo db
            with st.spinner('Uploading to MongoDB'):
                db_entries = []
                chunk_index = 0
                for chunk_text, chunk_embedding in zip(chunks, chunk_embeddings):
                    db_entries.append({
                        "_id": f"{st.session_state.file.name}:{chunk_index}",
                        "text":  chunk_text,
                        "embedding": chunk_embedding.tolist(),
                        "metadata": {
                            "file": st.session_state.file.name
                        }
                    })
                    chunk_index += 1
                self.mongodb = MongoDB(database_name="chatwpdf", collection_name='uploaded_docs')
                db_load_result = self.mongodb.load_chunks(db_entries)
                print(db_load_result)

            # create the vector search index on the uploaded data
            with st.spinner("Creating Search Index"):
                # create the vector search index
                self.mongodb.create_index(
                    index_name='vector_index',
                    dimensions=768,
                    similarity='cosine',
                    embedding_field='embedding'
                )

    def clear(self, item:str):
        # clear chat history or mongodb database
        if item == 'chat':
            st.session_state.messages=[]
            for x in st.session_state.messages:
                with st.chat_message(x['role']):
                    st.write(x['content'])
        elif item == 'collection':
            self.mongodb.database.drop_collection(self.mongodb.collection)


def main():
    App()

if __name__ == "__main__":
    # TODO
    main()
