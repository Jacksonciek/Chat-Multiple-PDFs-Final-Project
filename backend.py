import weaviate
from dotenv import load_dotenv
from weaviate.classes.config import Configure, Property, DataType
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from pypdf import PdfReader

load_dotenv()

client = weaviate.connect_to_local()

def store_pdf(pdf_file):
    embeddings = OpenAIEmbeddings()

    # Delete existing collections
    client.collections.delete_all()

    # Create new collection
    client.collections.create(
        name="Chatbot",
        description="Documents for chatbot",
        vectorizer_config=Configure.Vectorizer.text2vec_openai(
            model="ada",
            type_="text"
        ),
        properties=[
            Property(name="content", data_type=DataType.TEXT, description="Extracted text content"),
            Property(name="source", data_type=DataType.TEXT, description="Source type: pdf"),
            Property(name="filename", data_type=DataType.TEXT, description="PDF filename")
        ]
    )

    vectorstore = WeaviateVectorStore(
        client=client, index_name="Chatbot", text_key="content", embedding=embeddings
    )

    # Process single PDF file
    pdf_reader = PdfReader(pdf_file)
    text = ""

    for page in pdf_reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text + "\n"

    if not text.strip():
        return {"error": "No text could be extracted from the PDF"}

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_text(text=text)

    # Create metadata for all chunks
    metadatas = [{"source": "pdf", "filename": pdf_file.filename}] * len(chunks)

    # Store in vector database
    vectorstore.add_texts(chunks, metadatas=metadatas)

    return {"message": f"PDF '{pdf_file.filename}' successfully stored in the vector database"}

def query_chatbot(question):
    llm = OpenAI()
    embeddings = OpenAIEmbeddings()

    vectorstore = WeaviateVectorStore(
        client=client, index_name="Chatbot", text_key="content", embedding=embeddings
    )

    # Simple similarity search without memory context
    docs = vectorstore.similarity_search(question, k=4)

    if not docs:
        return {"answer": "I couldn't find relevant information in the uploaded PDF to answer your question."}

    # Generate answer using QA chain
    read_chain = load_qa_chain(llm=llm)
    answer = read_chain.run(input_documents=docs, question=question)
    
    return {"answer": answer}