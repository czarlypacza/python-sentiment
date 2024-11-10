from flask import Flask, jsonify, request
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

@app.route('/retrain', methods=['POST'])
def retrain_endpoint():
    train_model()
    return jsonify({"message": "Model retrained and reloaded successfully"}), 200

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
    app.run(host='0.0.0.0', port=5001, threaded=True)  # Runs the Flask server accessible from other devices