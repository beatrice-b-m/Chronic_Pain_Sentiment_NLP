#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
import matplotlib as mpl
import matplotlib.pyplot as plt

# Path to your Tinos-Regular.ttf file
font_path = '/Users/ipsayadav/Desktop/emory/BMI550_NLP/Tinos-Regular.ttf'  # Replace with your actual file path

# Add the custom font to Matplotlib's font manager
mpl.font_manager.fontManager.addfont(font_path)

# Set the font family to your custom font
mpl.rc('font', family='Tinos')
# Load data
data_beatrice = pd.read_csv('subset_iaa_out_bea_modified.csv')
data_ipsa = pd.read_csv('iosa_subset.csv')
data_peibo = pd.read_csv('subset_iaa_out_peibo.csv')

# Assign names for clarity
data_beatrice.name = 'Beatrice'
data_ipsa.name = 'Ipsa'
data_peibo.name = 'Peibo'

# Column containing the annotations
# category = 'general_sentiment'
category = 'provider_sentiment'

# Function to calculate Cohen's Kappa Score
def calculate_kappa(dataframe1, dataframe2, category):
    annotations1 = dataframe1[category].values
    annotations2 = dataframe2[category].values
    kappa = cohen_kappa_score(annotations1, annotations2)
    return pd.DataFrame([[dataframe1.name, dataframe2.name, kappa]], columns=['Annotator1', 'Annotator2', 'Kappa Score'])

# Calculate kappa scores for all pairs
kappa_beatrice_ipsa = calculate_kappa(data_beatrice, data_ipsa, category)
kappa_beatrice_peibo = calculate_kappa(data_beatrice, data_peibo, category)
kappa_ipsa_peibo = calculate_kappa(data_ipsa, data_peibo, category)

# Combine all kappa scores into a single DataFrame
all_kappa_scores = pd.concat([kappa_beatrice_ipsa, kappa_beatrice_peibo, kappa_ipsa_peibo])

# Pivot for heatmap
pivoted_kappa_scores = all_kappa_scores.pivot(index='Annotator1', columns='Annotator2', values='Kappa Score')

sns.heatmap(pivoted_kappa_scores, annot=True, fmt=".3f", cmap="viridis", 
            linewidths=.5, cbar_kws={'label': 'Agreement Level'}, 
            annot_kws={"size": 14})  # You can change the size value as needed

# # Creating the heatmap
# plt.figure(figsize=(12, 8))
# sns.heatmap(pivoted_kappa_scores, annot=True, fmt=".3f", cmap="viridis", linewidths=.8, cbar_kws={'label': 'Agreement Level'})
# plt.title("Cohen's Kappa Scores Heatmap : Provider Sentiment", fontsize=18)
# plt.xlabel("Annotator", fontsize=18)
# plt.ylabel("Annotator", fontsize=18)
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
# plt.tight_layout()  # Adjust layout to fit all elements
# plt.show()

