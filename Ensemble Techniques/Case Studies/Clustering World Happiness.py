#!/usr/bin/env python3
"""
Hands-On Ensemble Learning with Python (Packt Publishing) Chapter 13: 
Clustering World Happiness
"""
import time
import numpy as np
import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)
import matplotlib.pyplot as plt
import openensembles
from termcolor import colored
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
colours = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan', 'grey', 'white']

start = time.perf_counter()
#%% Load The World Happiness Report Dataset
WHR2019_df = pd.read_csv('Data/World Happiness Report/2019/WHR 2019.csv')
regions2018_df = pd.read_csv('Data/World Happiness Report/2018/Regions.csv')

#%% Data Exploration and Analysis
#### Scatter Plots Between The Life Ladder And Different Factors 
print(colored('\u2500'*41 + ' Data Exploration ' + '\u2500'*41, 'blue', 
              attrs = ['bold']))
plt.figure(figsize = (15, 10))
for i, column in enumerate(WHR2019_df.columns[3:]):
    plt.subplot(3, 4, i+1)
    plt.scatter(WHR2019_df[column], WHR2019_df[WHR2019_df.columns[2]])
    plt.title(column)
    plt.ylabel(WHR2019_df.columns[2])   
plt.tight_layout()

#### Pearson's Correlation Coefficient (R) Of Each Factor To The Life Ladder
correlation_coefficients = np.array(WHR2019_df[WHR2019_df.columns[2:]].corr()
                                    ['Life Ladder'].drop('Life Ladder'))
correlation_df = pd.DataFrame({'Factors': WHR2019_df.columns[3:], 
                               'Correlation Coefficient (r)': correlation_coefficients})
print(correlation_df, '\n')

#### Number Of Countries Surveyed Each Year (2005-2018)
years_sorted = np.sort(WHR2019_df['Year'].unique())
num_countries_per_year = []
for year in years_sorted:
    num_countries_per_year.append(WHR2019_df[WHR2019_df['Year'] == year]
                                  ['Country'].count())
num_countries_by_year_df = pd.DataFrame(
    {'Year': years_sorted, 'Number of Countries': num_countries_per_year})
print(num_countries_by_year_df)

#### Distribution Of Countries From Each Region In Terms Of A Pie Chart
regions2018_df.groupby('Region Indicator').count().plot.pie(
    subplots = True, autopct = '%1.0f%%', figsize = (20, 15), legend = False)
plt.title('Distribution of Countries From Each Region')
plt.ylabel('')

#### Variations Of The Life Ladder Over The Years
WHR2019_df.boxplot(column = 'Life Ladder', by = 'Year', grid = False, figsize = (20, 15))
plt.suptitle('')
plt.title('Life Ladder By Year')

# Using the same set of countries surveyed in 2005 
WHR2005_df = WHR2019_df[WHR2019_df['Country'].isin(
    WHR2019_df[WHR2019_df['Year'] == 2005]['Country'].values)]
WHR2005_df.boxplot(column = 'Life Ladder', by = 'Year', grid = False, figsize = (20, 15))
plt.suptitle('')
plt.title('Life Ladder By Year Based Only On Countries Surveyed In 2005')

#%% Creating The Clustering Ensemble Using OpenEnsembles
# Using 2017 data since less data is missing in that year
WHR2017_df = WHR2019_df[WHR2019_df['Year'] == 2017]
WHR2017_df = WHR2017_df.fillna(WHR2017_df.median())
factors = WHR2017_df.columns[3:]

print(colored('\u2500'*34 + ' Clustering Using OpenEnsembles ' + '\u2500'*34, 'green', 
              attrs = ['bold']))

#### Using The Original Data
WHR2017_cluster_data = openensembles.data(WHR2017_df[factors], factors)
np.random.seed(123456)
n_clusters = []
ensemble_sizes = []
silhouette_scores = []
for K in [2, 4, 6, 8, 10, 12, 14]:
    for size in [5, 10, 20, 50]:
        ensemble = openensembles.cluster(WHR2017_cluster_data)
        for j in range(size):
            name = 'kmeans_{}_{}'.format(size, j)
            ensemble.cluster('parent', 'kmeans', name, K)
        n_clusters.append(K)
        ensemble_sizes.append(size)
        
        predictions = ensemble.finish_co_occ_linkage(threshold = 0.5)
        silhouette_scores.append(silhouette_score(
            WHR2017_df[factors], predictions.labels['co_occ_linkage']))
clustering_df = pd.DataFrame({'K': n_clusters, 'Ensemble Size': ensemble_sizes, 
                              'Silhouette Score': silhouette_scores})
clustering_crosstab = pd.crosstab(index = clustering_df['K'], 
                                  columns = clustering_df['Ensemble Size'],
                                  values = clustering_df['Silhouette Score'], 
                                  aggfunc = lambda x: x)
clustering_crosstab = clustering_crosstab.round(decimals = 3)
print(colored('Original:', 'yellow', attrs = ['bold']))
print(clustering_crosstab)

#### Using The Normalized Data
WHR2017_df_normalized = (
    WHR2017_df[factors] - WHR2017_df[factors].mean())/WHR2017_df[factors].std()
WHR2017_cluster_data_normalized = openensembles.data(WHR2017_df_normalized, factors)
np.random.seed(123456)
silhouette_scores_normalized = []
for K in [2, 4, 6, 8, 10, 12, 14]:
    for size in [5, 10, 20, 50]:
        ensemble_normalized = openensembles.cluster(WHR2017_cluster_data_normalized)
        for k in range(size):
            name = 'kmeans_{}_{}'.format(size, k)
            ensemble_normalized.cluster('parent', 'kmeans', name, K)
        
        predictions_normalized = ensemble_normalized.finish_co_occ_linkage(
            threshold = 0.5)
        silhouette_scores_normalized.append(silhouette_score(
            WHR2017_df_normalized, predictions_normalized.labels['co_occ_linkage']))
clustering_df_normalized = pd.DataFrame(
    {'K': n_clusters, 'Ensemble Size': ensemble_sizes, 
     'Silhouette Score': silhouette_scores_normalized})
clustering_crosstab_normalized = pd.crosstab(
    index = clustering_df_normalized['K'], 
    columns = clustering_df_normalized['Ensemble Size'],
    values = clustering_df_normalized['Silhouette Score'], aggfunc = lambda x: x)
clustering_crosstab_normalized = clustering_crosstab_normalized.round(decimals = 3)
print(colored('\nNormalized:', 'magenta', attrs = ['bold']))
print(clustering_crosstab_normalized)

#### Using The Transformed Data Obtained From t-SNE Decomposition 
WHR2017_df_transformed = pd.DataFrame(TSNE().fit_transform(WHR2017_df[factors]))
WHR2017_cluster_data_transformed = openensembles.data(WHR2017_df_transformed, [0, 1])
np.random.seed(123456)
silhouette_scores_transformed = []
for K in [2, 4, 6, 8, 10, 12, 14]:
    for size in [5, 10, 20, 50]:
        ensemble_transformed = openensembles.cluster(WHR2017_cluster_data_transformed)
        for k in range(size):
            name = 'kmeans_{}_{}'.format(size, k)
            ensemble_transformed.cluster('parent', 'kmeans', name, K)
        
        predictions_transformed = ensemble_transformed.finish_co_occ_linkage(
            threshold = 0.5)
        silhouette_scores_transformed.append(silhouette_score(
            WHR2017_df_transformed, predictions_transformed.labels['co_occ_linkage']))
clustering_df_transformed = pd.DataFrame(
    {'K': n_clusters, 'Ensemble Size': ensemble_sizes, 
     'Silhouette Score': silhouette_scores_transformed})
clustering_crosstab_transformed = pd.crosstab(
    index = clustering_df_transformed['K'], 
    columns = clustering_df_transformed['Ensemble Size'],
    values = clustering_df_normalized['Silhouette Score'], aggfunc = lambda x: x)
clustering_crosstab_transformed = clustering_crosstab_transformed.round(decimals = 3)
print(colored('\nTransformed Using t-SNE Decomposition:', 'cyan', attrs = ['bold']))
print(clustering_crosstab_transformed)

#%% Gaining Insights Into The Structure and Relationships Within The Dataset
clustering_ensemble = openensembles.cluster(WHR2017_cluster_data_transformed)
np.random.seed(123456)
for l in range(20):
    name = 'kmeans_{}-tsne'.format(l)
    clustering_ensemble.cluster('parent', 'kmeans', name, 10)
# Create the labels for the clusters
clustering_predictions = clustering_ensemble.finish_co_occ_linkage(threshold = 0.5)
# Include the labels for the clusters into the WHR2017 dataframe
WHR2017_df['Cluster'] = clustering_predictions.labels['co_occ_linkage']
factors_plus_life_ladder = np.insert(factors, 0, WHR2017_df.columns[2])
WHR2017_df_groupby_cluster = WHR2017_df.groupby('Cluster').mean()[
    factors_plus_life_ladder]

def create_barplot(column, nrows, ncols, index):
    plt.subplot(nrows, ncols, index + 1)
    column_values = WHR2017_df_groupby_cluster.sort_values('Life Ladder')[column]
    column_values.plot(kind = 'bar', title = column, 
                       ylim = [column_values.min()*0.95, column_values.max()*1.05],
                       color = plt.cm.tab10(range(len(factors_plus_life_ladder))))
    plt.tight_layout()

plt.figure(figsize = (15, 10))
for m, column in enumerate(factors_plus_life_ladder):
    create_barplot(column, 3, 4, m)

stop = time.perf_counter()
print('\u2550'*100)
duration = stop - start
minutes = divmod(duration, 60)[0]
seconds = divmod(duration, 60)[1]
print(colored('Execution Duration: {:.2f}s ({:.1f}mins, {:.2f}s)'.format(
    duration, minutes, seconds), 'red'))