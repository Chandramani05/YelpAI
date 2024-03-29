from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from prompt import get_prompt
from langchain.chains import ConversationalRetrievalChain

import os
from reviews import upload_embeddings
from restraunts import get_restaurant_info
from dotenv import load_dotenv

load_dotenv()
def get_chroma_client():
    embedding_function = OpenAIEmbeddings(api_key=os.environ['OPEN_API_KEY'])
    return Chroma(
        collection_name="yelp_data",
        embedding_function=embedding_function,
        persist_directory="data/chroma")


def store_docs(docs):
    vector_store = get_chroma_client()
    vector_store.add_documents(docs)


def make_chain():
    model = ChatOpenAI(api_key=os.environ['OPEN_API_KEY'],
            model_name="gpt-3.5-turbo",
            temperature=0.3,
            verbose=False
        )
    vector_store = get_chroma_client()
    prompt = get_prompt()

    retriever = vector_store.as_retriever(search_type="mmr", verbose=True)

    chain = ConversationalRetrievalChain.from_llm(
        model,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs=dict(prompt=prompt),
        verbose=False,
        rephrase_question=False,
    )
    return chain


def get_response_chat(question, organization_name):
    chat_history = ""
    chain = make_chain()
    question  = "The question is reagrding : " + organization_name +  "And question is : " + question
    response = chain({"question": question, "chat_history": chat_history,
                      "organization_name": organization_name,
                      })
    return response['answer']


def save_vector(id) :
    rest = get_restaurant_info(str(id))
    store_docs(upload_embeddings(rest))
    vector_store = get_chroma_client()
    vector_store.get(include=['embeddings','metadatas','documents'])
    
def get_response (question, name) :
    response = get_response_chat(question, name)
    return response

