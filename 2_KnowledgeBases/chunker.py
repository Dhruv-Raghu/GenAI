# RegEx to split text
import re
# Sentence Transformers for quick embedding model
from sentence_transformers import SentenceTransformer
# similarity to comapare embeddings
from sentence_transformers.util import cos_sim
import numpy as np
import pymupdf

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

    def chunk(self, doc):
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
                    'file':doc.name
                }
                sentences.append(sentence_data)

        print("\nAdding Buffer...")
        # add n sentences (based on buffer size) before and after each sentence as context
        # added to a new key 'sentence_with_context'
        self.combine_sentences(sentences=sentences)

        print("\nGenerating Embeddings...")
        # generate embeddings for each sentence
        self.generateEmbeddings(sentences)
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
            chunk_id = f"{group[0]['file']}:{chunk_index}:{chunk_page}"
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
            chunk_id = f"{group[0]['file']}:{chunk_index}:{chunk_page}"
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

        return chunks


def main():
    chunker = SemanticChunker(bufferSize=1, breakpointPercentile=70)
    print('Opening Doc...')
    doc = pymupdf.open('data/neural_vision.pdf')
    # doc = pymupdf.open('/Users/dhruv/Documents/Y4S2/FYP_Sem1/Wachaja2015 - Navigating Blind People with a Smart Walker.pdf')
    chunks = chunker.chunk(doc)

    for chunk in chunks:
        print('\n\n')
        print(chunk)


    # sem_chunker = SemanticChunker()
    # sem_chunker.chunk(pdfData=data)


if __name__ == '__main__':
    main()
