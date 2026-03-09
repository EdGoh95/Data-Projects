#!/usr/bin/env python3
"""
Machine Learning Infrastructure And Best Practices For Software Engineers (Packt Publishing)
Chapter 7: Featureb Engineering for Numerical and Image Data
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

ant_df = pd.read_excel("../Chapter 6 - Processing Data In Machine Learning Systems/Regenerated PROMISE and BPD Datasets.xlsx",
                       sheet_name = 'ant_1_3', index_col = 0)
ant_features = ant_df.drop(['Defect'], axis = 1)

#%% Principal Component Analysis (PCA)
PCA_decomp = PCA(n_components = 3)

PCA_ant_features = PCA_decomp.fit_transform(ant_features)

plt.figure(figsize = (10, 10))
sns.set_style('white')
sns.set_palette('rocket')
sns.barplot(x = ['PC 1', 'PC 2', 'PC 3'], y = PCA_decomp.explained_variance_ratio_, palette = 'rocket')
plt.xlabel('Principal Components (PC)')
plt.ylabel('% Variance Explained')

#%% t-student distribution Stochastic Network Embedding (t-SNE)
TSNE_transform = TSNE(n_components = 3)
TSNE_ant_features = TSNE_transform.fit_transform(ant_features)

TSNE_first_comp, TSNE_second_comp, TSNE_third_comp = np.split(TSNE_ant_features,
                                                              indices_or_sections = 3, axis = 1)
TSNE_scatter = px.scatter_3d(TSNE_ant_features, np.ndarray.flatten(TSNE_first_comp),
                             np.ndarray.flatten(TSNE_second_comp), np.ndarray.flatten(TSNE_third_comp),
                             color = ant_df['Defect'].astype(str), color_discrete_sequence = ['red', 'green'],
                             labels = dict(x = 'First Component', y = 'Second Component',
                                           z = 'Third Component', color = 'No. of Defects'))
TSNE_scatter.update_scenes(xaxis = dict(gridcolor = 'black'), yaxis = dict(gridcolor = 'black'),
                           zaxis = dict(gridcolor = 'black'))
TSNE_scatter.write_html('t-SNE Transformation With 3 Components.html')

#%% Independent Component Analysis (ICA)
ICA_transform = FastICA(n_components = 3)
ICA_ant_features = ICA_transform.fit_transform(ant_features)
ICA_first_comp, ICA_second_comp, ICA_third_comp = np.split(ICA_ant_features, indices_or_sections = 3,
                                                           axis = 1)
ICA_scatter = px.scatter_3d(ICA_ant_features, np.ndarray.flatten(ICA_first_comp),
                            np.ndarray.flatten(ICA_second_comp), np.ndarray.flatten(ICA_third_comp),
                            color = ant_df['Defect'].astype(str), color_discrete_sequence = ['red', 'green'],
                            labels = dict(x = 'First Component', y = 'Second Component',
                                          z = 'Third Component', color = 'No. of Defects'))
ICA_scatter.update_scenes(xaxis = dict(gridcolor = 'black'), yaxis = dict(gridcolor = 'black'),
                          zaxis = dict(gridcolor = 'black'))
ICA_scatter.write_html('ICA Transformation With 3 Components.html')

#%% Locally Linear Embedding (LLE)
LLE_transform = LocallyLinearEmbedding(n_components = 3)
LLE_ant_features = LLE_transform.fit_transform(ant_features)
LLE_first_comp, LLE_second_comp, LLE_third_comp = np.split(LLE_ant_features, indices_or_sections = 3,
                                                           axis = 1)
LLE_scatter = px.scatter_3d(LLE_ant_features, np.ndarray.flatten(LLE_first_comp),
                            np.ndarray.flatten(LLE_second_comp), np.ndarray.flatten(LLE_third_comp),
                            color = ant_df['Defect'].astype(str), color_discrete_sequence = ['red', 'green'],
                            labels = dict(x = 'First Component', y = 'Second Component',
                                          z = 'Third Component', color = 'No. of Defects'))
LLE_scatter.update_scenes(xaxis = dict(gridcolor = 'black'), yaxis = dict(gridcolor = 'black'),
                           zaxis = dict(gridcolor = 'black'))
LLE_scatter.write_html('Locally Linear Embedding With 3 Components.html')

#%% Latent Discriminant Analysis (LDA)
LDA = LinearDiscriminantAnalysis(n_components = 1).fit(ant_features, ant_df['Defect'])
LDA_ant = LDA.transform(ant_features)