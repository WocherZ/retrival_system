import warnings
warnings.filterwarnings('ignore')

import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models.gigachat import GigaChat


GIGACHAT_TOKEN = "Y2VhYTI1YzctMGNjOS00YmIzLWFlNGQtODMzMjYyYTFmM2UwOjlhOTFkODA2LTAxMTgtNDBhYS05OWY3LTQxZDc1MzUxNDYxZA=="
llm = GigaChat(credentials=GIGACHAT_TOKEN, model='GigaChat', verify_ssl_certs=False)

st.title("Вопросно-ответная система на основе языковой модели")

uploaded_file = st.file_uploader("Загрузить свой документ в формате txt", type=["txt"], )

if uploaded_file:
    loader = TextLoader('./docs/' + uploaded_file.name, encoding='utf-8')
    documents = loader.load()

    splitter = MarkdownTextSplitter(chunk_size=400, chunk_overlap=0)
    texts = splitter.split_documents(documents)

    embeddings = SentenceTransformerEmbeddings(model_name="distiluse-base-multilingual-cased")
    vector_store = Chroma.from_documents(texts, embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={'k': 5})
    )

query = st.text_input("Введите свой вопрос")

if st.button("Получить ответ на основе системы поиска по документам"):
    if uploaded_file:
        answer = qa.run(query)
        st.write(answer)
    else:
        st.write("Сначала загрузите документ!")