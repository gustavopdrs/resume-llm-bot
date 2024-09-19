from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
import torch

# Ensure torch uses CPU
torch.set_default_device('cpu')
INDEX_NAME = "langchain-pinecone"


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = HuggingFaceEmbeddings(model_name="sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    from langchain_ollama import ChatOllama 
    chat = ChatOllama(model="llama3")

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    new_result = {
        "query":result["input"],
        "result":result["answer"],
        "source_documents": result["context"],
    }
    return new_result


if __name__ == "__main__":
    res = run_llm(query="Quem é Gustavo Pedroso Gal")
    print(res["result"])