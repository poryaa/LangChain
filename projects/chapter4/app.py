from flask import Flask, request, jsonify, render_template
from llm_core import answer_question

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")  # serve the web UI

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Missing 'question'"}), 400
    answer = answer_question(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
