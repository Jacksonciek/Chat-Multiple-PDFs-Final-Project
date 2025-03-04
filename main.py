from flask import Flask, request, jsonify
from backend import store_pdfs, query_chatbot

app = Flask(__name__)

# In-memory storage for session-based memory
conversation_memory = {}

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "files" not in request.files:
        return jsonify({"error": "No file part"}), 400

    files = request.files.getlist("files")  
    if not files or any(f.filename == "" for f in files):
        return jsonify({"error": "No selected files"}), 400

    response = store_pdfs(files) 
    return jsonify(response)

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    session_id = data.get("session_id")
    question = data.get("question")

    if not question or not session_id:
        return jsonify({"error": "Missing session_id or question"}), 400

    # Initialize session memory if not exists
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []

    # Retrieve past interactions
    chat_history = conversation_memory[session_id]

    # Get chatbot response
    answer = query_chatbot(question, chat_history)

    # Store the question-answer pair in session memory
    conversation_memory[session_id].append({"question": question, "answer": answer["answer"]})

    return jsonify({"session_id": session_id, "answer": answer["answer"]})

if __name__ == "__main__":
    app.run(debug=True)
