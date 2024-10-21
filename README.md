# GenAI: Documenting my Generative AI journey
As a fresh graduate with an interest in ML and AI concepts, the new wave of Generative AI and LLMs peaked my interest. With a strong base in the fundamentals of ML, Neural Networks and some basic AI concepts, I have decided to explore the world of Generative AI through real world projects and document my learning journey.
I will be using AWS, more specifically Amazon Bedrock since I was lucky enough to get access to some AWS credits through a hackathon.

PS: I am not a professional so take everything with a grain of salt.

## Prerequisites
- [x] Install and configure the AWS CLI (https://tinyurl.com/2b63f584)
- [x] Install boto3
- [x] Langchain_aws (optional package that integrates LLM features with AWS services. Documentation is messy, might be better to stick with using boto3 since the functionality that langchain provides can still be achieved without the package)
- [x] Lanchain/ langchain_core/ langchain_community (Don't get me started on this mess, but this is a rant for another time and place)
- [x] Streamlit (for the web app)

## 1. Text Generation
This is where I start my journey with Generative AI. The main aim behind this chapter is to get my feet wet with the basics of generative AI.
### [1.1. translate.py](1_Text_Generation/translate.py)
My very first generative AI project. The aim is to create a simple script that takes an english phrase and uses an LLM to translate it to another language.
#### Learning Outcomes
- Understanding boto3
- Configure the LLM and its paramters.
- Use langchain_aws to invoke responses.
- Zero-shot prompting
### [1.2. translate_app.py](1_Text_Generation/translate_app.py)
Create a chat app using Streamlit (my first time using streamlit) that allows the user to select a language and input a phrase to be translated.
#### Learning Outcomes
- Create a simple web app for the translator using Streamlit.
- Create and use a chat history to store previous translations.

## 2. Knowledge Bases
Hallucinations are a common issue in generative ai and occur when models generate incorrect or misleading results. This is usually due to a lack of relevant context. Knowledge Bases help combat this issue by prioviding models with the required information and context. Knowledge Bases are also useful in Question Answering tasks or in business situations where the model needs to be aware of certain facts.

RAG (Retrieval-Augemented Generation) is a popular technique to make LLMs more effective. In RAG, an embedding model is used to convert documents and other relevant information into a vector store. This vector store is used as a knowledge base which is then used to provide the LLM with more relevant context.

### [2.1. chatpdf.py](2_KnowledgeBases/chatpdf.py)
Create a chat with pdf application that uses RAG to provide the LLM with the correct context while answering quetions about the pdf. I want to avoid using Langchain inorder to learn exactly what is happening at each stage of the application.

#### Challenges
There are 2 main challenges with this project

1. PDF documents are hard to interpret correctly
2. Finding the optimal chunking strategy that can split the document into smaller chunks while still maintaing contextual information

To tackle the first problem, I dove head first into the huge collection of pdf libraries and parsers. After trying out way too many packages and libraries (pypdf, pdfminer, pdfplumber, unstructured, etc.), i narrowed it down to 2 good options. Marker-pdf and PyMuPDF. Marker-pdf is a more  well rounded pdf parser. It is able to extract text, headers, images, tables, equations in some cases and convert pdf documents in markdown. Markdown documents are easier to use with LLMs compared to PDFs. It worked really well, but the main issue was that it was too computationally heavy to run. Each pdf took way too long to parse on my laptop and as a result could not be used for this application.

I found PyMuPDF to be much faster but more tailored to my use case. I only needed text from the document which PyMuPDF was able to do very well, even in cases of scientific papers with 2 colmns of text. The only issue with PyMuPDF (and other pdf libraries) is that breaks are added after each line as opposed to the end of a paragraph. This can affect the chunking strategy as an ideal place to chunk is at the end of each paragraph which can be identified by a break. in the end, this was somewhat dealt with using the regular expressions python library by only keeping line breaks that came after punctuation marks. Which would indicate the end of a paragraph (in most situations).

The 2nd problem was finding an optimal chunking strategy. Instead of using basic chunking strategies like chunking after a certain number of tokens which won't be able to group together relevant information, I opted for a more dynamic and flexible chunking strategy. I created a Semantic Chunker from scratch that splits the text into sentences and then groups sentences together based on semantic similarity of sentence embeddings. This worked relatively well, but there was one major issue. The created chunks had no contextual information from other chunks or paragraphs in the text. This reduced the performance of retrievals when trying to identify chunks that are similar to the query. 

The solution to this problem came with a novel chunking strategy by Jina AI called Late Chunking for Long Context Embedding Models. The idea behind this is to embed the entire document first using a long context embedding model (something with an input of 8000 tokens or more should be enough). The token embeddings should have contextual information from the entire document this way. These token embeddings can then be grouped into chunks (hence the name late chunking). There was a significant improvement in retrieving information and the relevant chunks after switching to the late chunking method. Here is a link to the Github Repo for Jina AI's late chunking strategy: https://github.com/jina-ai/late-chunking

#### Learning Outcomes
- Text extraction from pdfs
- Semantic Chunking (from scratch)
- Late Chunking (better chunking strategy)
- Storing embeddings in vector database (MongoDB using PyMongo library)
- Vector Search using MongoDB Atlas Search

#### Future Work
- Adding more metadata to chunks (page numbers, etc.) so sources can be added to the response
- Process more information from PDFs like Images and Equations to be used as context.
- Application doesn't do too well with summarization tasks as only 3 relevant chunks are returned per query.

_Work in progress..._
