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

    response = store_pdfs(files) 
    return jsonify(response)

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "Missing question"}), 400

    answer = query_chatbot(question)
    return jsonify({"answer": answer["answer"]})

if __name__ == "__main__":
    app.run(debug=True)