import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

class RAGManager:
    def __init__(self, embedding_model_api_key, chunk_size=1000, chunk_overlap=100):
        # Inisialisasi embedding model and text splitter
        self.embedding_model = OpenAIEmbeddings(api_key=embedding_model_api_key, model="text-embedding-3-large")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    def load_and_embed_pdfs(self, pdf_files):
        """Memuat PDF, membagi dokumen menjadi beberapa bagian, and generate embeddings."""
        documents = {}
        split_docs = {}
        embedded_docs = {}

        # Load, split, and embed setiap PDF
        for title, file_path in pdf_files.items():
            print(f"Loading {title} from {file_path}...")

            # Memuat konten PDF
            loader = PyPDFLoader(file_path)
            documents[title] = loader.load()

            # Membagi dokumen menjadi beberapa bagian
            split_docs[title] = self.text_splitter.split_documents(documents[title])

            # Generate embeddings untuk setiap bagian
            print(f"Generating embeddings for {title}...")
            embeddings = self.embedding_model.embed_documents([chunk.page_content for chunk in split_docs[title]])

            # Menyimpan teks dan pasangan embeddings
            embedded_docs[title] = list(zip([chunk.page_content for chunk in split_docs[title]], embeddings))
        
        return embedded_docs

    def store_embeddings_in_faiss(self, embedded_docs):
        """Menyimpan embeddings dokumen di FAISS"""
        for title, text_embedding_pairs in embedded_docs.items():
            print(f"Storing embeddings for {title} in FAISS...")
            
            # Buat penyimpanan dan menyimpan vektor FAISS
            vector_store = FAISS.from_embeddings(text_embedding_pairs, self.embedding_model)
            vector_store.save_local(f"db_{title.lower().replace(' ', '_')}")

    def load_faiss_db(self, material_choice):
        """Memuat penyimpanan vektor FAISS untuk materi yang dipilih"""
        print(f"Loading FAISS database for {material_choice}...")
        vector_store = FAISS.load_local(
            f"db_{material_choice.lower().replace(' ', '_')}", 
            self.embedding_model,
            allow_dangerous_deserialization=True
            )
        return vector_store

    def query_faiss(self, material_choice, query):
        """Menanyakan penyimpanan vektor FAISS untuk dokumen yang relevan."""
        # Load the correct FAISS store
        vector_store = self.load_faiss_db(material_choice)
        
        # Retrieve relevant documents
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        retrieved_docs = retriever.invoke(query)
        
        return retrieved_docs

if __name__ == "__main__":
    pdf_files = {
        "Peristiwa Rengasdengklok": "docs/Peristiwa Rengasdengklok & Proklamasi.pdf",
        "Peristiwa 10 Nopember": "docs/Peristiwa 10 Nopember.pdf",
        "Konferensi Meja Bundar": "docs/Konferensi Meja Bundar.pdf"
    }

    rag_manager = RAGManager(embedding_model_api_key=os.getenv("EMBED_API_KEY"))
    embedded_docs = rag_manager.load_and_embed_pdfs(pdf_files)
    rag_manager.store_embeddings_in_faiss(embedded_docs)