from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://brainlox.com/courses/category/technical")

documents = loader.load()
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=7000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
from langchain.vectorstores import Chroma
db = Chroma.from_documents(docs, embedding_function)
query = "Python Playground"
docs = db.similarity_search(query)

# print results
print(docs[0].page_content)
db2 = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")
docs = db2.similarity_search(query)
db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
docs = db3.similarity_search(query)
print(docs[0].page_content)