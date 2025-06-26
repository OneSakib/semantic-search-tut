from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI  # or ChatOpenAI


# Initialize embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
# Load vector store
vectorstore = FAISS.load_local("product_vectorstore", embedding_model)
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

response = qa_chain.run("Which iPhone has best camera?")
print(response)
