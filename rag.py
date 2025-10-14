import streamlit as st
import os.path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

'''
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
如果需要使用OpenAI密钥对 请解除这部分注释 并将42-47行部分阿里云的llm和embedding加载部分注释掉
load_dotenv('/Users/kane/PycharmProjects/rag_demo/.env')
openai_endpoint: str = os.getenv('OPENAI_ENDPOINT')
openai_api_key: str = os.getenv('OPENAI_API_KEY')
openai_api_version: str = os.getenv('OPENAI_API_VERSION')
openai_deployment: str = os.getenv('OPENAI_DEPLOYMENT')
embedding_deployment: str = os.getenv('EMBEDDING_DEPLOYMENT')
embedding_api_version: str = os.getenv('EMBEDDING_API_VERSION')
embedding_api_key: str = os.getenv('EMBEDDING_API_KEY')
embedding_endpoint: str = os.getenv('EMBEDDING_ENDPOINT')
llm = ChatOpenAI(
    deployment=openai_deployment,
    openai_api_version=openai_api_version,
    endpoint=openai_endpoint,
    api_key=openai_api_key,
)
embeddings = OpenAIEmbeddings(
    openai_api_version=embedding_api_version,
    base_url=embedding_endpoint,
    api_key=embedding_api_key,
    deployment=embedding_deployment
 )


from langchain_community.embeddings import DashScopeEmbeddings
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=ALI-API-KEY)
llm = ChatOpenAI(base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
                 api_key="ALI-API-KEY",
                 model="qwen2.5-72b-instruct", temperature=0.7)
'''

from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai  import ChatVertexAI

GOOGLE_PROJEXT_ID = "mars-first-rag"

embeddings = VertexAIEmbeddings(
    project=GOOGLE_PROJEXT_ID,
    location="us-central1",
    model_name="gemini-embedding-001"
)
llm = ChatVertexAI(
    project=GOOGLE_PROJEXT_ID,
    location="us-central1",
    model_name="gemini-2.5-pro",
    temperature=0.7,
)


def text_chunk(file_path):
    # 加载指定路径的文本文件
    loader = TextLoader(file_path, encoding='utf-8')
    docs = loader.load()
    print(docs[0].metadata)

    # 把文本分割成 500 字一组的切片
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50  # 设置文本重叠
    )
    chunks = text_splitter.split_documents(docs)
    return chunks


def chunk2vector(docs, embeddings):
    # new_client = chromadb.EphemeralClient()
    vector = FAISS.from_documents(
        documents=docs,  # 设置保存的文档
        embedding=embeddings  # 设置 embedding model
        )
    return vector




def llm_chain(vector):
    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 

    Question: {question} 

    Context: {context} 

    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    retriever = vector.as_retriever()
    chain = (
            RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
            | prompt
            | llm
            | StrOutputParser()
    )
    return chain


def llm_an(file_path, question):
    # 避免question输入为空导致报错
    if not question:
        question = "hello"
    docs = text_chunk(file_path)
    vetcor = chunk2vector(docs, embeddings)
    chain = llm_chain(vetcor)
    answer = chain.invoke(question)
    return answer
