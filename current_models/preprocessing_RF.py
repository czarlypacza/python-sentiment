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

# Sample 4000 reviews from each original class
num_samples_to_keep = 7000
for star in ['1', '2', '4', '5']:
    global_data[star] = random.sample(global_data[star], min(num_samples_to_keep, len(global_data[star])))

# Sample 8000 reviews from class 3
global_data['3'] = random.sample(global_data['3'], min(10000, len(global_data['3'])))

# Convert to 3 classes
converted_data = {'1': [], '2': [], '3': []}

# Combine classes 1-2 into new class 1
for review in global_data['1'] + global_data['2']:
    review['rating'] = '1'  # negative
    converted_data['1'].append(review)

# Class 3 becomes new class 2
for review in global_data['3']:
    review['rating'] = '2'  # neutral
    converted_data['2'].append(review)

# Combine classes 4-5 into new class 3
for review in global_data['4'] + global_data['5']:
    review['rating'] = '3'  # positive
    converted_data['3'].append(review)

# Save converted data
with open('./global_data_3class.json', 'w') as json_file:
    json.dump(converted_data, json_file)

# Plot distribution
import matplotlib.pyplot as plt

class_counts = [len(converted_data[str(i)]) for i in range(1, 4)]

plt.bar(['Negative (1)', 'Neutral (2)', 'Positive (3)'], class_counts)
plt.xlabel('Rating Class')
plt.ylabel('Count')
plt.title('Label Distribution (3 Classes)')
plt.savefig('./plots/label_distribution_3class.png')
plt.clf()
