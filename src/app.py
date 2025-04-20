from detoxify import Detoxify
from flask import Flask,request,jsonify
from transformers import pipelines
app = Flask(__name__)
model = Detoxify("original")
summarizer = pipelines.pipeline("summarization", model="facebook/bart-large-cnn")
sentiment = pipelines.pipeline("sentiment-analysis",model="cardiffnlp/twitter-roberta-base-sentiment")
@app.route("/analyze",methods=["POST"])
def analyze():
    data=request.get_json();
    text = data.get("text","")
    results = model.predict(text)
    results = {key: float(value) for key, value in results.items()}
    return jsonify(results)

@app.route("/summarize",methods=["POST"])
def summarize():
    data = request.get_json()
    text = data.get("text","")
    results = summarizer(text,max_length=280,min_length=30,do_sample=False)
    return jsonify({"summary":results[0]['summary_text']})


@app.route("/sentiment",methods=["POST"])
def sentiment_analysis():
    data = request.get_json()
    text = data.get("text")
    result = sentiment(text)
    return jsonify(result[0])   


if __name__ == "__main__":
    app.run(port=5000)
