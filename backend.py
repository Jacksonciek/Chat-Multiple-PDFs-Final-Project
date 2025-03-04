import os
import weaviate
from weaviate import WeaviateClient
from weaviate import connect
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

# WEAVIATE_URL = os.environ["WCD_DEMO_URL"]
# WEAVIATE_API_KEY = os.environ["WCD_DEMO_RO_KEY"]
# auth_config = weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY)
# client = weaviate.Client(url=Config.WEAVIATE_URL, auth_client_secret=None)

load_dotenv()

client = weaviate.connect_to_local(
        skip_init_checks=True
    )

def store_pdfs(pdf_files):
    embeddings = OpenAIEmbeddings()

    client.collections.delete_all()

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

    all_texts = []  
    all_metadatas = [] 

    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        text = ""

        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"

        if not text.strip():
            continue

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = text_splitter.split_text(text=text)

        all_texts.extend(chunks)
        all_metadatas.extend([{"filename": pdf.filename}] * len(chunks))

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
