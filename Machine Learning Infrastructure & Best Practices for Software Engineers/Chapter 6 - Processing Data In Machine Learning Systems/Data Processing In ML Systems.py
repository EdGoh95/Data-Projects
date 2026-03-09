#!/usr/bin/env python3
"""
Machine Learning Infrastructure And Best Practices For Software Engineers (Packt Publishing)
Chapter 6: Processing Data in Machine Learning (ML) Systems
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from datasets import load_dataset

#%% Numerical Data Processing
ant_df = pd.read_excel('Regenerated PROMISE and BPD Datasets.xlsx', sheet_name = 'ant_1_3',
                       index_col = 0)

#### Summarizing & Visualizing Numerical Data
ant_correlogram = sns.pairplot(ant_df, diag_kind = 'kde')
plt.figure(figsize = (20, 15))
ant_heatmap = sns.heatmap(ant_df, cmap = 'Reds')
plt.tight_layout()

# KDE/Density Plots
sns.set_style('white')
fig, axes = plt.subplots(1, 2)
sns.kdeplot(x = ant_df['WMC'], y = ant_df['ImportCoupling'], cmap = 'Reds', fill = True,
            bw_adjust = 0.5, ax = axes[0])
sns.kdeplot(x = ant_df['NOM'], y = ant_df['DCC'], cmap = 'Reds', fill = True, bw_adjust = 0.5,
            ax = axes[1])
# Scatter Plots (Bubble Diagrams)
plt.figure()
sns.scatterplot(data = ant_df, x = 'NOM', y = 'DCC', size = 'Defect', legend = False,
                sizes = (20, 2000))
# Boxplots For Summarizing Individual Measures
with plt.style.context(style = 'seaborn-v0_8'):
    plt.figure()
    sns.boxplot(x = ant_df['Defect'], y = ant_df['CBO'], palette = 'Set2',
                flierprops = {'marker': 'd', 'markerfacecolor': 'k', 'markersize': 6})
    plt.figure()
    sns.violinplot(x = 'Defect', y = 'CBO', data = ant_df, palette = 'Set1')

#### PCA - Reducing The Number of Measures In Numerical Data
ant_features = ant_df.drop(['Defect'], axis = 1)
PCA_decomp = PCA(n_components = 2, random_state = 42)
principal_comp = PCA_decomp.fit_transform(ant_features)

# Visualizing The Principal Components
PC1, PC2 = np.split(principal_comp, 2, axis = 1)
plt.figure(figsize = (20, 20))
scatter = plt.scatter(PC1, PC2, c = ant_df['Defect'], cmap = ListedColormap(['red', 'darkgreen']),
                      alpha = 0.3)
plt.title('Plot of The First 2 Principal Components (PC 1 & 2)')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(*scatter.legend_elements(), title = 'No. of Defects', alignment = 'left')

#%% Image Data Processing
food101_dataset = load_dataset('food101', split = 'train')
plt.figure(figsize = (20, 10))
sns.histplot(data = food101_dataset['label'])

#%% Text Data Processing
imdb_dataset = load_dataset('imdb')
plt.figure()
sns.histplot(imdb_dataset['train']['label'], bins = 2)

code2text_dataset = load_dataset('code_x_glue_ct_code_to_text', 'java', split = 'test')
code_tokens_list = [item for sublist in code2text_dataset['code_tokens'] for item in sublist]
print(code_tokens_list[:10])

code_tokens_df = pd.DataFrame(code_tokens_list, columns = ['Token'])
code_tokens_counts = code_tokens_df.groupby('Token').size().reset_index(name = 'Count')
code_tokens_counts = code_tokens_counts.sort_values(by = 'Count', ascending = False)
plt.figure(figsize = (30, 15))
sns.barplot(x = 'Token', y = 'Count', data = code_tokens_counts[:25],
            palette = sns.color_palette('BuGn_r', n_colors = 25))
plt.title('Top 25 Tokens By Count')
plt.xticks(rotation = 45, horizontalalignment = 'right')