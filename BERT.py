import pandas as pd
from transformers import BertTokenizer, BertModel, Trainer, TrainingArguments
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Load dataset
myDataset = pd.read_json("results_new.json")
myDataset = myDataset[myDataset['sentence'] != '']
myDataset = myDataset.reset_index(drop=True)
print(myDataset.head())
print(myDataset.shape)


from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load pre-trained model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True)

tokenized_data = myDataset.apply(tokenize_function, axis=1)

def make_prediction(input_text):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.softmax(logits, dim=-1)

    # Extract positive and negative sentiment scores
    pos_score = predictions[0][1].item()
    neg_score = predictions[0][0].item()

    return {'positive': pos_score, 'negative': neg_score}

from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        inputs = tokenizer(item['sentence'], padding='max_length', truncation=True, return_tensors='pt')
        inputs = {key: val.squeeze() for key, val in inputs.items()}
        inputs['labels'] = torch.tensor([item['pos'], item['neg']])
        return inputs

if __name__ == '__main__':

    dataset = SentimentDataset(myDataset)

    # Split the dataset into training and evaluation datasets
    train_data, eval_data = train_test_split(myDataset, test_size=0.2)

    # Create evaluation dataset
    eval_dataset = SentimentDataset(eval_data)

    train_dataset = SentimentDataset(train_data)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=2,  # Reduced epochs to speed up training
        per_device_train_batch_size=32,  # Increased batch size to speed up training
        per_device_eval_batch_size=32,  # Increased batch size to speed up evaluation
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        save_steps=1000,  # Save checkpoints more frequently
        logging_steps=500,  # Log training metrics every 500 steps
        evaluation_strategy='steps',  # Evaluate every 500 steps
        fp16=True,  # Use mixed precision training for faster training
        dataloader_num_workers=6,  # Utilize all CPU cores for data loading
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()

    trainer.evaluate()

    # Save the fine-tuned model and tokenizer
    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')

    # Load the fine-tuned model and tokenizer
    model_path = './fine_tuned_model'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)

    # Loop to get user input and make predictions
    loop = True
    while loop:
        input_text = input("Please enter your input text: ")
        if input_text.lower() == "exit":
            loop = False
            break
        prediction = make_prediction(input_text)
        print(prediction)

