from flask import Flask, jsonify
from src.chat import LLM

llm = LLM(path=".models/Llama-3.2-1B", device=0)
app = Flask(__name__)


@app.route('/chat')
def chat():
    test_prompt="How does a plant feed itself?"
    respone = llm.chat(test_prompt)
    return jsonify({"response":respone})

if __name__ == '__main__':
    app.run(debug=True)
