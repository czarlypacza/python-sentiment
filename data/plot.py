import matplotlib.pyplot as plt
import numpy as np

metrics = ['Accuracy', 'Precision (Positive)', 'Precision (Negative)', 'Recall (Positive)', 'Recall (Negative)', 'F1 Score (Positive)', 'F1 Score (Negative)']
standard_results = [0.8037, 0.748, 0.892, 0.916, 0.691, 0.823, 0.779]
twitter_results = [0.6706, 0.636, 0.730, 0.799, 0.542, 0.708, 0.622]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, standard_results, width, label='Standard')
bars2 = ax.bar(x + width/2, twitter_results, width, label='Twitter')

ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Metrics between Standard and Twitter Datasets')
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.show()