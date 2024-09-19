from os import getenv
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
import asyncio
from langchain_pinecone import PineconeEmbeddings
from langchain.document_loaders import TextLoader


def ingest_docs2() -> None:
    embeddings = HuggingFaceEmbeddings(model_name="sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja")
    from langchain_community.document_loaders import FireCrawlLoader

    langchain_documents_base_urls = [
        "https://www.linkedin.com/in/gustavo-pedroso-gal-08b4bb1bb/",
        "https://www.linkedin.com/in/gustavo-pedroso-gal-08b4bb1bb/details/certifications/",
        "https://www.linkedin.com/in/gustavo-pedroso-gal-08b4bb1bb/details/skills/",

    ]
    langchain_documents_base_urls2 = [langchain_documents_base_urls[0]]
    for url in langchain_documents_base_urls2:
        print(f"FireCrawling {url=}")
        loader = FireCrawlLoader(
            url=url,
            mode="crawl",
            params={
                "limit": 5,
            },
        )
        docs = loader.load()

        print(f"Going to add {len(docs)} documents to Pinecone")
        PineconeVectorStore.from_documents(
            docs, embeddings, index_name="firecrawl-index"
        )
        print(f"****Loading {url}* to vectorstore done ***")

def ingest_docs():
  embeddings = HuggingFaceEmbeddings(model_name="sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja")
 # loader = ReadTheDocsLoader(encoding="UTF-8", path="gustavo-docs/")

  loader = TextLoader("gustavo-docs/gustavo.txt", encoding="UTF-8")
  raw_documents = loader.load()
  print(f"loaded {len(raw_documents)} documents")
  print(f"Loaded document content: {raw_documents[0].page_content}")

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
  documents = text_splitter.split_documents(raw_documents)
  print(f"Documents after splitting: {documents}")
  for doc in documents:
    print("it entered the loop")
    new_url = doc.metadata["source"]
    print(new_url)
    new_url = new_url.replace("langchain-docs", "https:/")
    doc.metadata.update({"source": new_url})

  print(f"Going to add {len(documents)} to Pinecone")
  PineconeVectorStore.from_documents(
     documents, embeddings, index_name="langchain-pinecone"
  )


async def main():
  ingest_docs2()


if __name__ == "__main__":
  asyncio.run(main())