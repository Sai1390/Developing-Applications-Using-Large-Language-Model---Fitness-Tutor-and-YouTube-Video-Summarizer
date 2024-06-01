import os
from langchain import PromptTemplate
from constants import openai_key
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.llms import openai
from langchain.embeddings.openai import OpenAIEmbeddings

import pinecone
from langchain.vectorstores import Pinecone

os.environ["OPENAI_API_KEY"] = openai_key
directory = 'data'

def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

documents = load_docs(directory)
print(len(documents))

def split_docs(documents, chunk_size=3000, chunk_overlap=300):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(documents)
print(len(docs))


embeddings = OpenAIEmbeddings()

query_result = embeddings.embed_query("Hello world")
print(len(query_result))


pinecone.init(
    api_key="2bd3886c-37cc-4599-8861-95f511351795",
    environment="us-central1-gcp"
)

index_name = "sai"

index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

def get_similiar_docs(query,k=1,score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query,k=k)
  else:
    similar_docs = index.similarity_search(query,k=k)
  return similar_docs


from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
model_name = "gpt-3.5-turbo"
temperature = 0.9  # Set the temperature value
max_tokens=20

llm = ChatOpenAI(model_name=model_name, temperature=temperature, max_tokens=max_tokens)



from langchain.chains.question_answering import load_qa_chain

chain = load_qa_chain(llm, chain_type="stuff")

def get_answer(query):
  similar_docs = get_similiar_docs(query)
  answer = chain.run(input_documents=similar_docs, question=query)
  return answer



import streamlit as st

st.title('Fitness Tutor')
input_text = st.text_input("Search Your Training Program ")


template = '''
          
Sure, here is another template:

Client: I'm looking for a workout program that will help me lose weight.

Fitness Tutor: Sure, I can help you with that. What are your fitness goals?

Client: I'd like to lose 10 pounds in the next 6 weeks.

Fitness Tutor: Okay, I can create a program that will help you achieve that goal. Here's what we'll do:

We'll start by assessing your current fitness level.
Then, we'll create a workout program that's tailored to your individual needs and goals.
We'll also provide you with nutritional guidance to help you make healthy food choices.
Client: That sounds great!

Fitness Tutor: I'm glad you think so. Let's get started!
          {context}
          Client: {human_input}
          Fitness Tutor:'''


prompt = PromptTemplate(input_variables=["context", "human_input"], template=template)

if input_text:
    documents = load_docs('data')
    docs = split_docs(documents)
    embeddings = OpenAIEmbeddings()
    index_name = "sai"
    index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    llm = OpenAI(model_name='gpt-3.5-turbo')
    

    chain = load_qa_chain(llm=llm, chain_type="stuff")
    with st.spinner('Generating Training Program...'):
        response = get_answer(input_text)
    st.write(response)




