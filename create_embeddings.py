from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Initialize embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
products = [
    {
        "title": "i-phone 16 brand new",
        "description": "Latest i-phone with great features",
        "price": "$999",
        "id": "prod_001"
    },
    {
        "title": "Samsung Galaxy S23",
        "description": "New Samsung flagship phone",
        "price": "$899",
        "id": "prod_002"
    },
]


# Convert product data to Langchain Documents
docs = [
    Document(
        page_content=prod["title"] + ". " + prod["description"],
        metadata={"id": prod["id"], "price": prod["price"]}
    )
    for prod in products
]

# Create vector store
vectorstore = FAISS.from_documents(docs, embedding_model)

# Save to disk if needed
vectorstore.save_local("product_vectorstore")
