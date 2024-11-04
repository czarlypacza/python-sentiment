from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import torch

app = Flask(__name__)
CORS(app)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model and tokenizer from the fine-tuned model path
model_path = '../python/regressor/my_awesome_model2/checkpoint-31500'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

model.to(device)
model.eval()

def make_prediction(input_text):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.softmax(logits, dim=-1)
    
    # Extract positive and negative sentiment scores
    pos_score = predictions[0][0].item()
    neg_score = predictions[0][1].item()
    return {'positive': pos_score, 'negative': neg_score}



@app.route('/sentiment/<name>', methods=['GET'])
def classify_reviews(name):
    result = []
    response = requests.get(f'http://localhost:5000/reviews/{name}')
    if response.status_code == 200:
        reviews = response.json()
        for review in reviews[name]:
            for sentence in review['text'].split('.'):
                if sentence.strip():
                    sentiment = make_prediction(sentence)
                    sentiment = [sentiment['positive'], -sentiment['negative']]
                    result.append({sentence: sentiment})
    else:
        return jsonify({"error": "Failed to fetch reviews"}), 500
    result = {'result': result}
    return jsonify(result)



@app.route('/sentiment', methods=['POST'])
def classify_review():
    data = request.get_json()
    input_text = data.get('text', '')

    if not input_text:
        return jsonify({"error": "No text provided"}), 400

    sentences = input_text.split('.')
    sentences = [sentence for sentence in sentences if sentence.strip()]
    results = []
    scores = []

    for sentence in sentences:
        if sentence.strip():
            sentiment = make_prediction(sentence)
            sentiment = [sentiment['positive'], -sentiment['negative']]
            pos, neg = sentiment
            
            sentiment_label = "positive" if pos > neg*-1 else "negative" if neg*-1 > pos else "neutral"
            results.append({
                "sentence": sentence,
                "pos": pos,
                "neg": neg,
                "sentiment": sentiment_label
            })
            scores.append({"pos": pos, "neg": neg})

    response = {
        "scores": scores,
        "results": results
    }
    print(response)

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)  # Runs the Flask server accessible from other devices