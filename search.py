from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Initialize embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
# Load vector store
vectorstore = FAISS.load_local("product_vectorstore", embedding_model)

# User input
query = "iphone 16"

# Perform semantic search
results = vectorstore.similarity_search(query, k=5)

# Show results
for res in results:
    print(res.metadata, res.page_content)
