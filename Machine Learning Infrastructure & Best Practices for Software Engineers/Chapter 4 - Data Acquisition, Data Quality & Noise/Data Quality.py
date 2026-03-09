#!/usr/bin/env python3
"""
Machine Learning Infrastructure And Best Practices For Software Engineers (Packt Publishing)
Chapter 4: Data Acquisition, Data Quality, and Noise
"""
import matplotlib.pyplot as plt
import pandas as pd

iq_problems_configuration_libraries = 0
iq_problems_entity_access = 0
try:
    for logMessage in open("Information Quality (Gerrit).log", 'r'):
        logItem = logMessage.split(';')
        logTime, logSource, logLevel , logProblem, _ = logItem

        if logLevel == 'ERROR':
            if 'LIBRARIES' in logProblem:
                iq_problems_configuration_libraries += 1
            elif 'ENTITY_ACCESS' in logProblem:
                iq_problems_entity_access += 1

except Exception as e:
    iq_general_error = 1

def getIndicatorColour(indicator_value):
    if indicator_value > 0:
        return 'Red'
    else:
        return 'Green'

rows = ['Entity Access Check', 'Libraries']
columns = ['Information Quality Check', 'Value']
cell_text = [['Entity Access: {}'.format(iq_problems_entity_access)],
             ['Libraries: {}'.format(iq_problems_configuration_libraries)]]
colours = [[getIndicatorColour(iq_problems_entity_access)],
           [getIndicatorColour(iq_problems_configuration_libraries)]]

fig, ax = plt.subplots()
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText = cell_text, cellColours = colours, colLabels = columns, loc = 'best',
                 fontsize = 18)
table.scale(1, 2)

#%% Noise
reviews_df = pd.read_csv('../Source Code From GitHub/chapter_4/book_chapter_4_embedded_1k_reviews.csv')
all_reviews_count = len(reviews_df['Summary'])
unique_reviews_count = len(reviews_df['Summary'].unique())
print('Number of duplicate entries (potential noisy data): {:d}\n'.format(
    all_reviews_count - unique_reviews_count))
grouped_reviews = reviews_df.groupby(by = reviews_df['Summary']).count()
duplicates = grouped_reviews[grouped_reviews['Time'] > 1].index.to_list()

for duplicate_review in duplicates:
    duplicate_df = reviews_df[reviews_df['Summary'] == duplicate_review]
    num_labels = len(duplicate_df['Score'].unique())
    if num_labels > 1:
        reviews_df.drop(reviews_df[reviews_df['Summary'] == duplicate_review].index, inplace = True)
        print('Duplicate review: {}, Number of Labels: {:d}'.format(
            duplicate_review, len(duplicate_df['Score'].unique())))