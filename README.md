# GenAI: Documenting my journey learning Generative AI concepts
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
### 1.1. [translate.py](1_Text_Generation/translate.py)
My very first generative AI project. The aim is to create a simple script that takes an english phrase and uses an LLM to translate it to another language.
#### Learning Outcomes
- Understanding boto3
- Configure the model and its paramters.
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
### 2.1. chatpdf.py
Create a model that uses RAG to answer questions based on a pdf document.
### Learning Outcomes
- Convert pdf document into a vector store using the Titan embeddings model.
- Retrieve relevant information from the vector store to augment the prompt and generate a response from the LLM.


_Work in progress..._
