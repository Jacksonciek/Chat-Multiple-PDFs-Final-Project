import os
import weaviate
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Weaviate
from weaviate.classes.init import Auth
import weaviate.classes as wvc
from langchain_weaviate import WeaviateVectorStore
from weaviate.classes.config import Configure, Property, DataType
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader

def comp_process(pdfs, question):
    load_dotenv()
    llm = OpenAI()
    wcd_url = os.environ["WCD_DEMO_URL"]
    wcd_api_key = os.environ["WCD_DEMO_RO_KEY"]
    
    text = ""
    
    # Read all PDFs
    for file in pdfs:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
            
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_text(text=text)
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    auth_config = weaviate.AuthApiKey(api_key=wcd_api_key)

    WEAVIATE_URL = wcd_url
    try: 
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=auth_config
        )
        
        client.collections.delete_all()
        client.collections.list_all()
        client.collections.create(
            name="Chatbot",
            description="Documents for chatbot",
            vectorizer_config=Configure.Vectorizer.text2vec_openai(
                model="ada",
                type_="text"
            ),
            properties=[
                Property(
                    name="content",
                    data_type=DataType.TEXT,
                    description="The content of the paragraph",
                    skip_vectorization=False,
                    vectorize_property_name=False
                )
            ]
        )

        vectorstore = WeaviateVectorStore(
            client=client,
            index_name="Chatbot",
            text_key="content",
            embedding=embeddings,
            attributes=["content"]
        )
        # Store chunks in Weaviate
        vectorstore.add_texts(chunks)

        # Perform similarity search
        docs = vectorstore.similarity_search(question, k=4)

        # Use OpenAI to generate answers
        read_chain = load_qa_chain(llm=llm)
        answer = read_chain.run(input_documents=docs, question=question)

        return answer
    
    finally:
        client.close()

# Query the model
answer = comp_process(["./tes.pdf"], "What is algorithm?")
print(answer)