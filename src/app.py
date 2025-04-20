from detoxify import Detoxify
from flask import Flask,request,jsonify
app = Flask(__name__)
model = Detoxify("original")
@app.route("/analyze",methods=["POST"])
def analyze():
    data=request.get_json();
    text = data.get("text","")
    results = model.predict(text)
    results = {key: float(value) for key, value in results.items()}
    return jsonify(results)
if __name__ == "__main__":
    app.run(port=5000)
