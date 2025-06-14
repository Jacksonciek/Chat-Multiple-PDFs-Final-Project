from flask import Flask, request, jsonify
from backend import store_pdf, query_chatbot

app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "files" not in request.files:
        return jsonify({"error": "No file part"}), 400

    files = request.files.getlist("files")  
    if not files or any(f.filename == "" for f in files):
        return jsonify({"error": "No selected files"}), 400

    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files are allowed"}), 400

    response = store_pdf(files) 
    return jsonify(response)

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    questions = data.get("questions")  # Sekarang menerima array pertanyaan

    if not questions or not isinstance(questions, list):
        return jsonify({"error": "Missing questions array or questions is not a list"}), 400

    answers = []
    for question in questions:
        if not question or not isinstance(question, str):
            answers.append({"question": question, "answer": "Invalid question format"})
            continue
        
        answer = query_chatbot(question)  # Tidak perlu session_id dan memory lagi
        answers.append({"question": question, "answer": answer["answer"]})

    return jsonify({"answers": answers})

if __name__ == "__main__":
    app.run(debug=True)