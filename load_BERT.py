from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Define the checkpoint directory
checkpoint_dir = './results/checkpoint-4000'  # Replace with the actual checkpoint directory

model_path = './fine_tuned_model'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(checkpoint_dir)

# Function to make predictions
def make_prediction(input_text):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.softmax(logits, dim=-1)
        return predictions

# Loop to get user input and make predictions
loop = True
while loop:
    input_text = input("Please enter your input text: ")
    if input_text.lower() == "exit":
        loop = False
        break
    prediction = make_prediction(input_text)
    print(prediction)
