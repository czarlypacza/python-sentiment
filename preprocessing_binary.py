import os
import json
import random

# Initialize data structure
global_data = {'1':[], '2':[], '3':[], '4':[], '5':[]}

# Load original data
for star in ['1', '2', '3', '4', '5']:
    folder_path = f'./training_data_stars_{star}'
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                global_data[star].extend(data)

# Set seed for reproducibility
random.seed(42)

# Sample equal amounts from each original class
num_samples_to_keep = 7000
for star in ['1', '2', '3', '4', '5']:
    global_data[star] = random.sample(global_data[star], min(num_samples_to_keep, len(global_data[star])))

# Convert to binary classes
converted_data = {'negative': [], 'positive': []}

# Combine classes 1-2 into negative
for review in global_data['1'] + global_data['2']:
    review['rating'] = 'negative'
    converted_data['negative'].append(review)

# Combine classes 3-5 into positive
for review in global_data['3'] + global_data['4'] + global_data['5']:
    review['rating'] = 'positive'
    converted_data['positive'].append(review)

# Save converted data
with open('./global_data_binary.json', 'w') as json_file:
    json.dump(converted_data, json_file)

# Plot distribution
import matplotlib.pyplot as plt

class_counts = [len(converted_data['negative']), len(converted_data['positive'])]

plt.bar(['Negative', 'Positive'], class_counts)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Label Distribution (Binary Classification)')
plt.savefig('./plots/label_distribution_binary.png')
plt.clf()