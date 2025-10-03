from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Carpeta donde se encuentran todos los PDFs
PDF_DIR = "./data/docs"
INDEX_DIR = "./faiss_index"

# Recolectar todos los PDFs
pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

all_documents = []

# Leer y cargar todos los PDFs
for pdf_file in pdf_files:
    loader = PyPDFLoader(os.path.join(PDF_DIR, pdf_file))
    docs = loader.load()
    all_documents.extend(docs)

# Dividir en fragmentos
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(all_documents)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Crear o cargar índice FAISS
if os.path.exists(INDEX_DIR):
    db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
else:
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(INDEX_DIR)

# Recuperador para usar en el chatbot
retriever = db.as_retriever(search_kwargs={"k": 3})  