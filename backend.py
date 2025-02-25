import os
import weaviate
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Weaviate
from weaviate.classes.init import Auth
import weaviate.classes as wvc
from langchain_weaviate import WeaviateVectorStore
from weaviate.classes.config import Configure, Property, DataType
from langchain_openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()
WEAVIATE_URL = os.environ["WCD_DEMO_URL"]
WEAVIATE_API_KEY = os.environ["WCD_DEMO_RO_KEY"]

# Initialize Weaviate connection
auth_config = weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY)
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=auth_config
)

def store_pdfs(pdf_files):
    """
    Process multiple PDFs and store them in Weaviate, ensuring each PDF is stored separately.
    """
    embeddings = OpenAIEmbeddings()

    # Delete the previous collection to prevent duplicate storage
    client.collections.delete_all()

    # Create a new collection for the chatbot knowledge base
    client.collections.create(
        name="Chatbot",
        description="Documents for chatbot",
        vectorizer_config=Configure.Vectorizer.text2vec_openai(
            model="ada",
            type_="text"
        ),
        properties=[
            Property(name="content", data_type=DataType.TEXT, description="Extracted text content"),
            Property(name="filename", data_type=DataType.TEXT, description="PDF filename")
        ]
    )

    vectorstore = WeaviateVectorStore(
        client=client, index_name="Chatbot", text_key="content", embedding=embeddings
    )

    all_texts = []  # Store extracted texts
    all_metadatas = []  # Store metadata (e.g., filenames)

    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        text = ""

        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"

        # Skip empty PDFs
        if not text.strip():
            continue

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = text_splitter.split_text(text=text)

        # Append extracted text and metadata for each PDF
        all_texts.extend(chunks)
        all_metadatas.extend([{"filename": pdf.filename}] * len(chunks))

    # Store all chunks in Weaviate
    vectorstore.add_texts(all_texts, metadatas=all_metadatas)

    return {"message": "All PDFs successfully stored in the vector database"}


def query_chatbot(question):
    llm = OpenAI()
    embeddings = OpenAIEmbeddings()

    vectorstore = WeaviateVectorStore(
        client=client, index_name="Chatbot", text_key="content", embedding=embeddings
    )

    docs = vectorstore.similarity_search(question, k=4)

    read_chain = load_qa_chain(llm=llm)
    answer = read_chain.run(input_documents=docs, question=question)

    return {"answer": answer}
