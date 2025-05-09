from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db")

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_ensemble_retriever(documents=None, persist_directory=CHROMA_PERSIST_DIRECTORY):
    os.makedirs(persist_directory, exist_ok=True)

    embeddings = get_embeddings()

    if documents:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=f"collection_{uuid.uuid4().hex[:8]}"
        )
        vectorstore.persist()
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 5
        vector_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 10,
                "lambda_mult": 0.7
            }
        )

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )

        return ensemble_retriever
    else:
        try:
            collections = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            ).get()

            if collections and len(collections['ids']) > 0:
                collection_name = collections['collection_name'][-1] if isinstance(collections['collection_name'], list) else collections['collection_name']

                vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=embeddings,
                    collection_name=collection_name
                )

                docs = []
                for i, _ in enumerate(collections['ids']):
                    if i < len(collections['metadatas']) and i < len(collections['documents']):
                        metadata = collections['metadatas'][i] if collections['metadatas'][i] else {}
                        docs.append(Document(
                            page_content=collections['documents'][i],
                            metadata=metadata
                        ))

                if docs:
                    bm25_retriever = BM25Retriever.from_documents(docs)
                    bm25_retriever.k = 5 
                    vector_retriever = vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": 5,
                            "fetch_k": 10,
                            "lambda_mult": 0.7
                        }
                    )

                    ensemble_retriever = EnsembleRetriever(
                        retrievers=[bm25_retriever, vector_retriever],
                        weights=[0.5, 0.5]
                    )

                    return ensemble_retriever
                else:
                    return vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": 5,
                            "fetch_k": 10,
                            "lambda_mult": 0.7
                        }
                    )
            else:
                return None
        except Exception as e:
            print(f"Error loading existing collections: {str(e)}")
            return None

def update_retriever_weights(retriever, bm25_weight=0.5, vector_weight=0.5):
    from langchain.retrievers import EnsembleRetriever
    if isinstance(retriever, EnsembleRetriever):
        retriever.weights = [bm25_weight, vector_weight]
    return retriever