import os
import json
import random

global_data = {'1':[], '2':[], '3':[], '4':[], '5':[]}

for star in ['1', '2', '3', '4', '5']:
    folder_path = f'./training_data_stars_{star}'
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                global_data[star].extend(data)
                

# Set a seed for reproducibility
random.seed(42)

# Determine the number of samples to keep
num_samples_to_keep = 4000  # Adjust this number as needed

# Randomly sample without replacement
global_data['5'] = random.sample(global_data['5'], min(num_samples_to_keep, len(global_data['5'])))
global_data['4'] = random.sample(global_data['4'], min(num_samples_to_keep, len(global_data['4'])))
global_data['3'] = random.sample(global_data['3'], min(num_samples_to_keep, len(global_data['3'])))
global_data['2'] = random.sample(global_data['2'], min(num_samples_to_keep, len(global_data['2'])))
global_data['1'] = random.sample(global_data['1'], min(num_samples_to_keep, len(global_data['1'])))
                
#print(global_data)
with open('./global_data.json', 'w') as json_file:
    json.dump(global_data, json_file)

import matplotlib.pyplot as plt

# Histograms for sores 1-5

star_counts = []

for star in ['1', '2', '3', '4', '5']:
    count = 0
    for entry in global_data[star]:
        count += 1
    star_counts.append(count)

plt.bar(['1', '2', '3', '4', '5'], star_counts)
plt.xlabel('Star Rating')
plt.ylabel('Count')
plt.title('Label distribution for all stars')
plt.savefig('./plots/label_distribution_all_stars.png')
plt.clf()  # Clear the figure for the next plot
