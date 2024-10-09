import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

class RAGManager:
    def __init__(self, embedding_model_api_key, chunk_size=1000, chunk_overlap=100):
        # Initialize embedding model and text splitter
        self.embedding_model = OpenAIEmbeddings(api_key=embedding_model_api_key, model="text-embedding-3-large")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    def load_and_embed_pdfs(self, pdf_files):
        """Load PDF files, split them into chunks, and generate embeddings."""
        documents = {}
        split_docs = {}
        embedded_docs = {}

        # Load, split, and embed each PDF
        for title, file_path in pdf_files.items():
            print(f"Loading {title} from {file_path}...")

            # Load PDF content
            loader = PyPDFLoader(file_path)
            documents[title] = loader.load()

            # Split the documents into chunks
            split_docs[title] = self.text_splitter.split_documents(documents[title])

            # Generate embeddings for each chunk
            print(f"Generating embeddings for {title}...")
            embeddings = self.embedding_model.embed_documents([chunk.page_content for chunk in split_docs[title]])

            # Store the text and embedding pairs
            embedded_docs[title] = list(zip([chunk.page_content for chunk in split_docs[title]], embeddings))
        
        return embedded_docs

    def store_embeddings_in_faiss(self, embedded_docs):
        """Store the document embeddings in FAISS."""
        for title, text_embedding_pairs in embedded_docs.items():
            print(f"Storing embeddings for {title} in FAISS...")
            
            # Create FAISS vector store and save it
            vector_store = FAISS.from_embeddings(text_embedding_pairs, self.embedding_model)
            vector_store.save_local(f"faiss_db_{title.lower().replace(' ', '_')}")

    def load_faiss_db(self, material_choice):
        """Load FAISS vector store for the selected material."""
        print(f"Loading FAISS database for {material_choice}...")
        vector_store = FAISS.load_local(f"faiss_db_{material_choice.lower().replace(' ', '_')}", self.embedding_model)
        return vector_store

    def query_faiss(self, material_choice, query):
        """Query the FAISS vector store for relevant documents."""
        # Load the correct FAISS store
        vector_store = self.load_faiss_db(material_choice)
        
        # Retrieve relevant documents
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        retrieved_docs = retriever.retrieve(query)
        
        return retrieved_docs

