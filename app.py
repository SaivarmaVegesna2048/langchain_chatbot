from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

app = Flask(__name__)
api = Api(app)

class DocumentSearch(Resource):
    def __init__(self):
        # Initialize the document loader
        loader = WebBaseLoader("https://brainlox.com/courses/category/technical")
        documents = loader.load()

        # Initialize the text splitter
        text_splitter = CharacterTextSplitter(chunk_size=7000, chunk_overlap=0)
        self.docs = text_splitter.split_documents(documents)

        # Initialize the embedding function
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # Create the Chroma database
        self.db = Chroma.from_documents(self.docs, self.embedding_function)
        self.db2 = Chroma.from_documents(self.docs, self.embedding_function, persist_directory="./chroma_db")
        self.db3 = Chroma(persist_directory="./chroma_db", embedding_function=self.embedding_function)

    def get(self):
        query = request.args.get('query')
        if not query:
            return {'error': 'Query parameter is required'}, 400

        # Perform the similarity search
        docs = self.db3.similarity_search(query)

        if not docs:
            return {'message': 'No documents found'}, 404

        return jsonify({'results': [doc.page_content for doc in docs]})

api.add_resource(DocumentSearch, '/search')

if __name__ == '__main__':
    app.run(debug=True)
