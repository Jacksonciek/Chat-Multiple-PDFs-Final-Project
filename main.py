from flask import Flask, request, jsonify
from backend import store_pdf, query_chatbot
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

app = Flask(__name__)

conversation_memory = {}  

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "files" not in request.files:
        return jsonify({"error": "No file part"}), 400

    files = request.files.getlist("files")  
    if not files or any(f.filename == "" for f in files):
        return jsonify({"error": "No selected files"}), 400

    response = store_pdf(files) 
    return jsonify(response)

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    session_id = data.get("session_id")
    question = data.get("question")

    if not question or not session_id:
        return jsonify({"error": "Missing session_id or question"}), 400

    if session_id not in conversation_memory:
        conversation_memory[session_id] = ConversationBufferMemory()

    memory = conversation_memory[session_id]

    answer = query_chatbot(question, session_id, memory)

    memory.chat_memory.add_message(HumanMessage(content=question))
    memory.chat_memory.add_message(AIMessage(content=answer["answer"]))

    return jsonify({"session_id": session_id, "answer": answer["answer"]})

if __name__ == "__main__":
    app.run(debug=True)
