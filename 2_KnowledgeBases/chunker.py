# RegEx to split text
import re
# Sentence Transformers for quick embedding model
from sentence_transformers import SentenceTransformer
from fmodels import TitanEmbeddings
import boto3
import os
# similarity to comapare embeddings
from sentence_transformers.util import cos_sim
import numpy as np

from transformers import AutoModel
from transformers import AutoTokenizer
import requests

class SemanticChunker():
    def __init__(self, bufferSize=1, breakpointPercentile=95) -> None:
        # how many sentences before and after to provide as context when creating embeddings
        self.bufferSize = bufferSize
        # percentile threshold to determine when to break a chunk
        self.breakpointPercentile = breakpointPercentile

    def sentence_splitter(self, text_data):
        # RegEx pattern splits text based on punctuation marks (.?!) followed by 1 or more whitespace
        sentence_list = re.split(r'(?<=[.?!])\s+', text_data)
        return sentence_list

    def combine_sentences(self, sentences):
        for i in range(len(sentences)):
            sentence_with_context = ''

            # add sentences before the current sentence based on buffer size
            for j in range(i-self.bufferSize, i):
                # if j < 0, there are not enough sentences before the current sentence to include as a buffer
                if j >= 0:
                    sentence_with_context += sentences[j]['text'] + ' '

            # add the current sentence
            sentence_with_context += sentences[i]['text']

            # add sentences after the current sentence based on buffer size
            for j in range(i+1, i+1+self.bufferSize):
                if j < len(sentences):
                    sentence_with_context += ' ' + sentences[j]['text']

            sentences[i]['sentence_with_context'] = sentence_with_context

    def generateEmbeddings(self, sentences):
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        for i in range(len(sentences)):
            sentences[i]['embedding'] = model.encode(sentences[i]['sentence_with_context'])


    def generateEmbeddingsTitan(self, sentences):
        bedrock_client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_DEFAULT_REGION", None))
        model = TitanEmbeddings(bedrock_client=bedrock_client)
        for i in range(len(sentences)):
            sentences[i]['embedding'] = model.generate_embeddings(sentences[i]['sentence_with_context'])[0]

    def calculate_distances(self, sentences):
        distances=[]
        for i in range(len(sentences)-1):
            current_embedding = sentences[i]['embedding']
            next_embedding = sentences[i+1]['embedding']

            similarity = cos_sim(current_embedding, next_embedding)
            distance = 1-(similarity[0].item())

            distances.append(distance)
            sentences[i]['distance_to_next'] = distance
        return distances, sentences

    def chunk(self, doc, file_name=None):
        if file_name is None:
            file_name='unnamed'

        sentences = []
        print("\nSplitting Sentences...")
        for page in doc:
            # print(page)
            text = page.get_textpage().extractTEXT()
            sentence_list = self.sentence_splitter(text_data=text)
            # add each sentence along with the page number as metadata
            for sentence in sentence_list:
                sentence_data = {
                    'text':sentence,
                    'page':page.number,
                    'file':file_name
                }
                sentences.append(sentence_data)
            # print(sentences[0]['file'])

        print("\nAdding Buffer...")
        # add n sentences (based on buffer size) before and after each sentence as context
        # added to a new key 'sentence_with_context'
        self.combine_sentences(sentences=sentences)

        print("\nGenerating Embeddings...")
        # generate embeddings for each sentence
        self.generateEmbeddingsTitan(sentences)
        # calculate cosine distances between embeddings
        distances, sentences = self.calculate_distances(sentences)

        print("\nChunking...")
        # chunk based on calculated breakpoints
        breakpoint_distance = np.percentile(distances, self.breakpointPercentile)
        breakpoint_indices = []
        # create a list of indices where the distance is greater than the threshold
        for i in range(len(distances)):
            if distances[i] > breakpoint_distance:
                breakpoint_indices.append(i)

        print("\nBreakpoint Indices: ")
        print(breakpoint_indices)

        # Start chunking based on breakpoint indices
        start_index = 0
        chunk_index = 0
        chunks = []
        for index in breakpoint_indices:
            end_index = index
            # group the sentences that will form a chunk
            group = sentences[start_index:end_index+1]
            chunk_text = ''
            chunk_page = group[0]['page']
            chunk_id = f"{file_name[0:-4]}:{chunk_index}:{chunk_page}"
            # add each sentence in the group to a string
            for sentence in group:
                chunk_text += (' ' + sentence['text'])
            # create a chunk and it to the list of chunks in json format
            chunks.append({
                "_id" : chunk_id,
                "text": chunk_text,
                "metadata":{
                    "page": chunk_page+1,
                    "file": group[0]['file']
                }
            })
            start_index = index+1
            chunk_index += 1
        # any remaining sentences after the last breakpoint index will form the final group
        if start_index < len(sentences):
            group = sentences[start_index:]
            chunk_text=''
            chunk_page = group[0]['page']
            chunk_id = f"{file_name[:-4]}:{chunk_index}:{chunk_page}"
            for sentence in group:
                chunk_text += (' ' + sentence['text'])

            chunks.append({
                "_id" : chunk_id,
                "text": chunk_text,
                "metadata":{
                    "page": chunk_page+1,
                    "file": group[0]['file']
                }
            })

        print("\nNumber of chunks: " + str(len(chunks)))

        for chunk in chunks:
            print('\n\n')
            print(chunk)

        return chunks

class LateChunker():
    def __init__(self):
        # load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
        self.model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)

    def jina_segmenter(self, text):
        # call the jina segmenter api
        url = 'https://segment.jina.ai/'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {os.environ.get('JINA', None)}'
        }
        data = {
            "content": text,
            "return_tokens": True,
            "return_chunks": True,
            "max_chunk_length": 1000
        }
        response_data = requests.post(url, headers=headers, json=data).json()
        chunks = response_data.get('chunks', [])
        chunk_positions = [(start, end) for start, end in response_data.get('chunk_positions', [])]
        # print(chunk_positions)

        # Unfortunately, chunk positions are based on character count rather than token count. We need the chunk positions with respect to tokens
        # Hence the following code:
        # tokenize the text
        inputs = self.tokenizer(text, return_tensors='pt', return_offsets_mapping=True)
        # return the start and end character indices for each token
        token_offsets = inputs['offset_mapping'][0].tolist()  # (start, end) positions for each token
        # the first and last items in the token offsets are (0,0) so we remove them
        token_offsets.pop(len(token_offsets)-1)
        token_offsets.pop(0)
        print(inputs)

        # create a list to store every token index where a chunk ends
        chunk_end_index = []
        # for each chunk get the start and end character indices
        for (chunk_start, chunk_end) in chunk_positions:
            # iterate through all the start and end character indices of the tokens
            for i, (start, end) in enumerate(token_offsets):
                # if it is the last token in the token list, add it to the list
                if i == len(token_offsets)-1:
                    chunk_end_index.append(i)
                # if the chunk end character position is lesser than the current token end position:
                elif chunk_end <= end:
                    # add the current token as the end of a chunk
                    chunk_end_index.append(i)
                    break   # break to move onto the next chunk end character position

        # now, populate the span annotations with the span of each chunk in terms of tokens
        span_annotations = []
        start = 0
        for index in chunk_end_index:
            span_annotations.append((start, index))
            start = index+1
        print(span_annotations)

        return chunks, span_annotations

    def late_chunking(self, model_output, span_annotations, max_length=None):
        # this function is adapted and repurposed from the jina ai late chunking repo: https://github.com/jina-ai/late-chunking
        token_embeddings = model_output[0]
        outputs=[]

        for embeddings, annotations in zip(token_embeddings, span_annotations):
            # remove annotations that go beyond the max length of the model
            if max_length is not None:
                annotations = [
                    (start, min(end, max_length - 1))
                    for (start, end) in annotations
                    if start < (max_length - 1)
                ]

            # mean pooling of chunk embeddings
            pooled_embeddings = [
                embeddings[start:end].sum(dim=0) / (end - start)
                for start, end in annotations
                if (end - start) >= 1
            ]

            # convert from tensor to python array
            pooled_embeddings = [
                embedding.detach().cpu().numpy() for embedding in pooled_embeddings
            ]
            outputs.append(pooled_embeddings)

        return outputs

    def get_chunk_embeddings(self, doc):
        print('extract text from pdf')
        text = ''
        for page in doc:
            text += page.get_textpage().extractTEXT()
        # remove unnecesary line breaks to clean up the text and chunk it better
        text = text.replace(' \n ', '')
        pattern = r"(?<![.!?])\n"
        text = re.sub(pattern, "", text)

        # chunk the text and return the start and end indices of the tokens of each chunk
        print('jina segmenter')
        chunks, span_annotations = self.jina_segmenter(text=text)

        # call the tokenizer on the input text and chunk token embeddings together based on the indices in the previous step
        print('late chunking')
        token_inputs = self.tokenizer(text, return_tensors='pt')
        model_output = self.model(**token_inputs)
        chunk_embeddings = self.late_chunking(model_output, [span_annotations], 8000)[0]

        # return the chunks and the chunk embeddings
        print('done hehe')
        return chunks, chunk_embeddings
