# !pip install pandas seaborn matplotlib
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load JSON data from file
with open('output/evaluation_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
# Convert JSON dict to list of dicts
records = []
for file_name, metrics in data.items():
    rec = {"file": file_name}
    rec.update(metrics)
    records.append(rec)

df = pd.DataFrame(records)

# Melt dataframe for seaborn
df_melted = df.melt(id_vars='file', value_vars=['recall', 'precision', 'f1'], var_name='Metric', value_name='Value')

plt.figure(figsize=(12,6))
sns.barplot(x='file', y='Value', hue='Metric', data=df_melted)
plt.title('Common Word Summary Evaluation Metrics')
plt.ylabel('Percentage')
plt.xlabel('File')
plt.show()
