#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os

# Get the CSV file path from the user
csv_file = input("Enter the path to the CSV file: ")

# Check if the specified file exists
if not os.path.isfile(csv_file):
    print("The specified CSV file does not exist.")
    exit(1)

# Read the CSV file
df = pd.read_csv(csv_file)

# Function to get user input for categorization
def categorize_manually(row):
    print(f"\n[Text {row['serial_number']}]\n{text}\n")
    while True:
        category = input("Enter category (1 for POSITIVE, 2 for NEGATIVE, 3 for NOT APPLICABLE, q to quit): ")
        if category in ('1', '2', '3', 'q'):
            return category
        else:
            print("Invalid input. Please enter 1, 2, 3, or q to quit.")

# Add a serial number column to the DataFrame
df['serial_number'] = range(1, len(df) + 1)

# Get the desired filename for the new CSV file from the user
new_csv_file = input("Enter the name for the new CSV file (e.g., output.csv): ")

# Iterate through each row in the DataFrame
while not df.empty:
    current_row = df.iloc[0]  # Get the first row
    text = current_row['text']

    # Manually categorize the text
    sentiment_category = categorize_manually(current_row)

    if sentiment_category == 'q':
        break  # Quit the categorization process

    # Append the data to the categorized_data list
    categorized_data = [current_row['serial_number'], current_row['screen_name'], text, current_row['tweet_id'], sentiment_category]

    # Create a new DataFrame with the categorized data
    categorized_df = pd.DataFrame([categorized_data], columns=['serial_number', 'screen_name', 'text', 'tweet_id', 'sentiment_category'])

    # Save the categorized data to the specified CSV file
    if not os.path.isfile(new_csv_file):
        categorized_df.to_csv(new_csv_file, index=False)
    else:
        categorized_df.to_csv(new_csv_file, mode='a', header=False, index=False)

    # Remove the categorized row from the original DataFrame
    df = df.iloc[1:]

print(f"Manually categorized data saved to {new_csv_file}")

