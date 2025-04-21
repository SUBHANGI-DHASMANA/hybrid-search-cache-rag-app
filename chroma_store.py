from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db")

def get_embeddings():
    return OllamaEmbeddings(model="all-minilm")

def create_ensemble_retriever(documents=None, persist_directory=CHROMA_PERSIST_DIRECTORY):
    os.makedirs(persist_directory, exist_ok=True)

    # Initialize embeddings
    embeddings = get_embeddings()

    if documents:
        # Create a new ChromaDB vector store
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=f"collection_{uuid.uuid4().hex[:8]}"
        )

        # Create BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 4

        vector_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        # Create ensemble retriever
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

                # Load the collection
                vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=embeddings,
                    collection_name=collection_name
                )

                # Get documents for BM25
                docs = []
                for i, doc_id in enumerate(collections['ids']):
                    if i < len(collections['metadatas']) and i < len(collections['documents']):
                        metadata = collections['metadatas'][i] if collections['metadatas'][i] else {}
                        docs.append(Document(
                            page_content=collections['documents'][i],
                            metadata=metadata
                        ))

                # Create BM25 retriever with fewer documents for speed
                bm25_retriever = BM25Retriever.from_documents(docs) if docs else None
                if bm25_retriever:
                    bm25_retriever.k = 4

                    vector_retriever = vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 4}
                    )

                    ensemble_retriever = EnsembleRetriever(
                        retrievers=[bm25_retriever, vector_retriever],
                        weights=[0.5, 0.5]
                    )

                    return ensemble_retriever
                else:
                    return vectorstore.as_retriever(search_kwargs={"k": 5})
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