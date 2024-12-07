# HSA_OVERRIDE_GFX_VERSION=10.3.0 /media/michal/dev1/sentiment/python/myenv/bin/python /media/michal/dev1/sentiment/python-sentiment/sentiment_BERT.py
#export HSA_OVERRIDE_GFX_VERSION=10.3.0 

import time
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import requests
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import torch
import os

app = Flask(__name__)
CORS(app)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load the model and tokenizer from the fine-tuned model path
model_path = '../python/regressor/my_awesome_model2/checkpoint-31500'
# Function to load the model and tokenizer
def load_model(model_path):
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

# Load the initial model
load_model(model_path)

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

# Function to scrape data and analyze sentiment
def scrape_and_analyze_data():
    # Placeholder for your scraping and sentiment analysis logic
    # Replace this with the actual implementation from your Next.js app
    data = []
    
    names = ["albinadyla.com","yourasteria.com","crystal classics","step","eloan","driveway", "Ruby Lane", "vanessaflair.com", "aardy", "seven corners","home zone furniture", "acitydiscount", "queensboro", "chrono24", "lost empire herbs","liftmode","suretybonds.com","trustage insurance agency","newday usa","figure","clever real estate","e-residence.com","rainbow shops","rotita","halara"]
    for name in names:
        response = requests.get(f'http://localhost:3000/api/rule?company={name}&limit=1500') 
        if response.status_code == 200:
            reviews = response.json()
            reviews = reviews['results']
            data.extend(reviews)
        else:
            print(f"Failed to fetch reviews for {name}")

    with open('./BERT/results_new_retrain.json', 'w') as f:
        json.dump(data, f, indent=4)

    return data

# Function to preprocess data
def preprocess_data(data):
    transformed_data = []
    for entry in data:
        if entry['sentence'].strip():
            transformed_entry = {
                'labels': [entry['pos'], -entry['neg']],
                'text': entry['sentence']
            }
            transformed_data.append(transformed_entry)
    return transformed_data

# Function to train the model
def train_model():
    data = scrape_and_analyze_data()
    transformed_data = preprocess_data(data)

    train_data, test_data = train_test_split(transformed_data, test_size=0.15, random_state=42)
    train_data, validate_data = train_test_split(train_data, test_size=0.15, random_state=42)

    with open('./BERT/train_data.json', 'w') as f:
        json.dump(train_data, f, indent=4)
    with open('./BERT/test_data.json', 'w') as f:
        json.dump(test_data, f, indent=4)
    with open('./BERT/validate_data.json', 'w') as f:
        json.dump(validate_data, f, indent=4)

    dataset = load_dataset('json', data_files={
        'train': './BERT/train_data.json',
        'test': './BERT/test_data.json',
        'validate': './BERT/validate_data.json'
    })

    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        r2 = r2_score(labels, predictions)
        return {"mse": mse, "mae": mae, "r2": r2}

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased",
        num_labels=2,
        problem_type="regression"
    ).to("cuda")

    training_args = TrainingArguments(
        output_dir="./BERT/my_awesome_model",
        logging_dir="./logs",
        logging_steps=100,
        learning_rate=2e-5,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_steps=1000,
        save_total_limit=2,
        save_steps=1000,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("./BERT/my_awesome_model")

    # Update the model path to the latest checkpoint
    global model_path
    model_path = "./BERT/my_awesome_model"

    # Reload the model
    load_model(model_path)

#export API_KEYS='ABCD-EFGH-IJKL-MNOP,QRST-UVWX-YZAB-CDEF,GHIJ-KLMN-OPQR-STUV,WXYZ-ABCD-EFGH-IJKL,MNOP-QRST-UVWX-YZAB'

def check_api_key(api_key):
    valid_api_keys = os.getenv('API_KEYS').split(',')
    return api_key in valid_api_keys

def requires_api_key(f):
    def decorated(*args, **kwargs):
        api_key = request.headers.get('x-api-key')
        if not api_key or not check_api_key(api_key):
            return Response('Unauthorized', 401)
        return f(*args, **kwargs)
    return decorated

@app.route('/retrain', methods=['POST'])
@requires_api_key
def retrain_endpoint():
    train_model()
    return jsonify({"message": "Model retrained and reloaded successfully"}), 200

@app.route('/sentiment/<name>', methods=['GET'])
def classify_reviews(name):
    limit = request.args.get('limit', default=None, type=int)
    result = []
    response = requests.get(f'http://localhost:5000/reviews/{name}?limit={limit}')
    if response.status_code == 200:
        reviews = response.json()
        for review in reviews[name]:
            for sentence in review['text'].split('.'):
                if sentence.strip():
                    sentiment = make_prediction(sentence)
                    sentiment = [sentiment['positive'], -sentiment['negative']]
                    result.append({sentence: sentiment})
    if response.status_code == 404:
        return jsonify({"error": "Company not found"}), 404
    if response.status_code != 200 and response.status_code != 404:
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

# Add these imports at the top if not present
import os
from pathlib import Path

# Add this endpoint to sentiment_BERT.py
@app.route('/analyze', methods=['PUT'])
def analyze_datasets():
    try:
        # Define data directory - adjust path as needed
        data_dir = Path('./data')
        
        # Analyze standard dataset
        standard_results = analyze_standard_dataset(data_dir)
        
        # Analyze Twitter dataset
        twitter_results = analyze_twitter_dataset(data_dir)
        
        # Analyze additional dataset
        additional_results = analyze_additional_dataset(data_dir)
        
        # Save results
        results = {
            "standardResults": standard_results,
            "twitterResults": twitter_results,
            "additionalResults": additional_results
        }
        
        with open(data_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        return jsonify(results), 200
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

def analyze_standard_dataset(data_dir):
    files = ['imdb_labelled.txt', 'yelp_labelled.txt', 'amazon_cells_labelled.txt']
    test_data = []
    
    for file in files:
        file_path = data_dir / file
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sentence, score = line.strip().split('\t')
                    sentiment = 'positive' if int(score) == 1 else 'negative'
                    test_data.append({"text": sentence, "sentiment": sentiment})
                    
    return calculate_metrics(test_data, "standard")

def analyze_twitter_dataset(data_dir):
    twitter_file = data_dir / 'test_twitter.csv'
    twitter_data = []
    
    with open(twitter_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                # Parse CSV line similar to TypeScript version
                parts = line.strip().split('","')
                if len(parts) >= 6:
                    sentiment = parts[0].replace('"', '')
                    text = parts[5].replace('"', '')
                    
                    # Map sentiment values
                    sentiment_map = {
                        '4': 'positive',
                        '2': 'neutral',
                        '0': 'negative'
                    }
                    mapped_sentiment = sentiment_map.get(sentiment.rstrip(','))
                    
                    if mapped_sentiment:
                        twitter_data.append({"text": text, "sentiment": mapped_sentiment})
                        
    return calculate_metrics(twitter_data, "twitter")

# Add this function to load and analyze the additional dataset
def analyze_additional_dataset(data_dir):
    additional_file = data_dir / 'train.jsonl'
    additional_data = []

    with open(additional_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line.strip())
                additional_data.append({"text": entry["text"], "sentiment": entry["label_text"]})

    return calculate_metrics(additional_data, "additional")

def calculate_metrics(test_data, dataset_type):
    true_positives = 0
    true_negatives = 0
    true_neutrals = 0
    false_positives = 0
    false_negatives = 0
    false_neutrals = 0  # Added this
    neutral_predictions = 0
    neutral_positive_misses = 0
    neutral_negative_misses = 0
    
    total = len(test_data)
    correct = 0
    
    # Determine if dataset includes neutral class
    neutral_in_dataset = any(item['sentiment'] == 'neutral' for item in test_data)
    
    # Start timing only predictions
    prediction_time = 0
    for item in test_data:
        start_time = time.time()
        prediction = make_prediction(item["text"])
        prediction_time += time.time() - start_time
        
        pos_score = prediction["positive"]
        neg_score = prediction["negative"]
        pos_score = round(pos_score, 4)
        neg_score = round(neg_score, 4)
        
        # Define a threshold for neutral classification
        threshold = 0.0001
        
        if abs(pos_score - 0.5) <= threshold and abs(neg_score - 0.5) <= threshold:
            predicted = "neutral"
        else:
            if pos_score > abs(neg_score):
                predicted = "positive"
            elif abs(neg_score) > pos_score:
                predicted = "negative"
        
        # predicted = "neutral"
        # if pos_score > abs(neg_score):
        #     predicted = "positive"
        # elif abs(neg_score) > pos_score:
        #     predicted = "negative"
            
        if predicted == "neutral":
            neutral_predictions += 1
            if item["sentiment"] == "positive":
                neutral_positive_misses += 1
                false_neutrals += 1
            elif item["sentiment"] == "negative":
                neutral_negative_misses += 1
                false_neutrals += 1
            elif item["sentiment"] == "neutral":
                correct += 1
                true_neutrals += 1
        elif predicted == item["sentiment"]:
            correct += 1
            if predicted == "positive":
                true_positives += 1
            else:
                true_negatives += 1
        else:
            if predicted == "positive":
                false_positives += 1
            else:
                false_negatives += 1

    accuracy = correct / total
    precision_pos = true_positives / (true_positives + false_positives or 1)  
    precision_neg = true_negatives / (true_negatives + false_negatives or 1)
    recall_pos = true_positives / (true_positives + false_negatives or 1)
    recall_neg = true_negatives / (true_negatives + false_positives or 1)
    
    f1_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos or 1)
    f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg or 1)

    # Calculate neutral metrics if neutral class is present
    if neutral_in_dataset:
        precision_neu = true_neutrals / (true_neutrals + false_neutrals or 1)
        recall_neu = true_neutrals / (true_neutrals + false_neutrals or 1)
        f1_neu = 2 * (precision_neu * recall_neu) / (precision_neu + recall_neu or 1)
    else:
        precision_neu = recall_neu = f1_neu = None

    result = {
        "datasetType": dataset_type,
        "timeTaken": int(prediction_time * 1000),
        "accuracy": accuracy,
        "precision": {
            "positive": precision_pos,
            "negative": precision_neg
        },
        "recall": {
            "positive": recall_pos,
            "negative": recall_neg
        },
        "f1Score": {
            "positive": f1_pos,
            "negative": f1_neg
        },
        "totalSamples": total,
        "correctPredictions": correct,
        "neutralStats": {
            "total": neutral_predictions,
            "missedPositives": neutral_positive_misses,
            "missedNegatives": neutral_negative_misses,
            "percentage": (neutral_predictions / total) * 100,
            "trueNeutrals": true_neutrals if neutral_in_dataset else None
        }
    }

    # Add neutral metrics if applicable
    if neutral_in_dataset:
        result["precision"]["neutral"] = precision_neu
        result["recall"]["neutral"] = recall_neu
        result["f1Score"]["neutral"] = f1_neu

    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)  # Runs the Flask server accessible from other devices