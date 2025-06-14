import weaviate
from dotenv import load_dotenv
from weaviate.classes.config import Configure, Property, DataType
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_weaviate import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from pypdf import PdfReader

load_dotenv()

client = weaviate.connect_to_local()

def store_pdfs(pdf_files):
    embeddings = OpenAIEmbeddings()

    client.collections.delete_all()

    client.collections.create(
        name="Chatbot",
        description="Documents and Conversations for chatbot",
        vectorizer_config=Configure.Vectorizer.text2vec_openai(
            model="ada",
            type_="text"
        ),
        properties=[
            Property(name="content", data_type=DataType.TEXT, description="Extracted text content"),
            Property(name="source", data_type=DataType.TEXT, description="Source type: 'pdf' or 'conversation'"),
            Property(name="filename", data_type=DataType.TEXT, description="PDF filename (if applicable)")
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
        all_metadatas.extend([{"source": "pdf", "filename": pdf.filename}] * len(chunks))

    vectorstore.add_texts(all_texts, metadatas=all_metadatas)

    return {"message": "All PDFs successfully stored in the vector database"}

def store_conversation(question, answer):
    embeddings = OpenAIEmbeddings()
    vectorstore = WeaviateVectorStore(
        client=client, index_name="Chatbot", text_key="content", embedding=embeddings
    )

    conversation_texts = [
        f"User: {question}",
        f"Chatbot: {answer}"
    ]

    conversation_metadata = [
        {"source": "conversation"},
        {"source": "conversation"}
    ]

    vectorstore.add_texts(conversation_texts, metadatas=conversation_metadata)

def query_chatbot(question, session_id, memory):
    llm = OpenAI()
    embeddings = OpenAIEmbeddings()

    vectorstore = WeaviateVectorStore(
        client=client, index_name="Chatbot", text_key="content", embedding=embeddings
    )

    chatbot_collection = client.collections.get("Chatbot")

    stored_messages = memory.chat_memory.messages
    context = " ".join([f"Q: {msg.content}" if isinstance(msg, HumanMessage) else f"A: {msg.content}" for msg in stored_messages])
    full_query = f"Context: {context} Current Question: {question}"

    search_results = chatbot_collection.query.near_vector(
        near_vector=embeddings.embed_query(full_query),
        distance=0.7,  
        limit=4  
    )

    docs = vectorstore.similarity_search(full_query, k=4)

    read_chain = load_qa_chain(llm=llm)
    answer = read_chain.run(input_documents=docs, question=question)

    store_conversation(question, answer)

    return {"answer": answer}
