#!/usr/bin/env python3
"""
Analysis of Iris Flower Dataset With Assistance From:
Mastering Machine Learning With Python In Six Steps Second Edition (Apress Media)
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import plotly.express as px
import argparse
from sklearn import metrics
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datafile', help = 'File name, together with its location. Data should be in the following format: Sepal Length (cm), Sepal Width (cm), Petal Length (cm), Petal Width (cm)')
    parser.add_argument('--input-parameters', help = 'Input should be in the following format: Sepal Length (cm), Sepal Width (cm), Petal Length (cm), Petal Width (cm)')
    return parser.parse_args()

def main(datafile, input_parameters):
    # Import and Load The Iris Dataset
    df_iris = pd.read_csv('Iris Dataset/iris.data', 
                          names = ['Sepal Length/cm', 'Sepal Width/cm', 
                                   'Petal Length/cm', 'Petal Width/cm', 
                                   'Species'])
    df_iris['Species'] = df_iris['Species'].str.replace('Iris-', '').str.capitalize()
    X = df_iris[['Sepal Length/cm', 'Sepal Width/cm', 
                 'Petal Length/cm', 'Petal Width/cm']]
    Y = df_iris['Species']

    # Scatter Plot of The Raw Data
    raw_plot = px.scatter_matrix(
        df_iris, dimensions = ['Sepal Length/cm', 'Sepal Width/cm', 
                               'Petal Length/cm', 'Petal Width/cm'],
        title = 'Pair Plot of Raw Data Features', color = df_iris['Species'], 
        opacity = 1.0)
    raw_plot.update_layout(title_x = 0.5, title_y = 0.95, 
                           titlefont = dict(size = 25), font = dict(size = 15), 
                           legend_font = dict(size = 14))
    raw_plot.update_traces(diagonal_visible = False)
    # raw_plot.write_html('Pair Plot of Raw Data.html')
    
    # Normalize The Dataset
    X = StandardScaler().fit_transform(X)

    # Split The Dataset Into Training and Testing Datasets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, 
                                                    random_state = 0)

    # Construct, Train and Evaluate The k-Nearest Neighbors (kNN) Model
    clf = KNeighborsClassifier(n_neighbors = 10, p = 2, metric = 'minkowski')
    clf.fit(X_train, Y_train)

    # Generate the evaluation metrics
    print('\u2500'*74 + ' Training ' + '\u2500'*74)
    print('Accuracy:', metrics.accuracy_score(Y_train, clf.predict(X_train)))
    print('Classification Report:')
    print(metrics.classification_report(Y_train, clf.predict(X_train)))

    print('\u2500'*74 + ' Testing ' + '\u2500'*74)
    print('Accuracy:', metrics.accuracy_score(Y_test, clf.predict(X_test)))
    print('Classification Report:')
    print(metrics.classification_report(Y_test, clf.predict(X_test)))
    if datafile:
        df_input = pd.read_csv(datafile, names = 
                      ['Sepal Length/cm', 'Sepal Width/cm', 'Petal Length/cm', 
                       'Petal Width/cm'])
    if input_parameters:
        input_parameters = input_parameters.split(',')
        df_input = pd.DataFrame([input_parameters], columns = 
                      ['Sepal Length/cm', 'Sepal Width/cm', 'Petal Length/cm', 
                       'Petal Width/cm'])
    
    column = []
    pos = []
    for i in range(len(df_input)):
        column.append(df_input.values[[i]])
        pos.append(df_iris[(df_iris['Sepal Length/cm'] == float(column[i][0][0]))
                           & (df_iris['Sepal Width/cm'] == float(column[i][0][1]))
                           & (df_iris['Petal Length/cm'] == float(column[i][0][2]))
                           & (df_iris['Petal Width/cm'] == float(column[i][0][3]))]
                   .index.tolist())
    pos_new = [index for index in range(len(pos)) if pos[index] == []]
    pos_exist = [position[0] for index, position in enumerate(pos) if pos[index] != []]
    
    if pos == []:
        # If the input parameters of the new flower(s) is not present in the 
        # Iris dataset, concatenate this new entry to the Iris dataframe
        df_iris = pd.concat([df_iris, df_input], ignore_index = True)
        X_update = df_iris[
            ['Sepal Length/cm', 'Sepal Width/cm', 'Petal Length/cm', 
             'Petal Width/cm']]
        X_update = StandardScaler().fit_transform(X_update)
        species_prediction = clf.predict(X)[
            [index for index in reversed(range(len(df_input.index)))]]
        print('Species (Predicted):', ', '.join(species_prediction))
        for j in reversed(range(len(df_input.index))):
            df_iris['Species'].loc[(len(df_iris.index)-1)-j] = species_prediction[j]
        nearest_neighbors = clf.kneighbors(X_update)[1]
        locations = []
        similiar_points = [[None for m in range(10)] for n in range(len(df_input.index))]
        for k in reversed(range(len(df_input.index))):
            locations.append(nearest_neighbors[(len(df_iris.index)-1)-k])
            similiar_points[k] = df_iris.loc[locations[k]]
        
        for l in range(len(similiar_points)):
            combined_df = pd.concat(
                [df_iris.loc[[(len(df_iris.index)-1)-l]], similiar_points[l]],
                ignore_index = True)
            similiar_points_plot = px.scatter_matrix(
                combined_df, dimensions = ['Sepal Length/cm', 'Sepal Width/cm', 
                                           'Petal Length/cm', 'Petal Width/cm'],
                color = combined_df['Species'], opacity = 1)
            similiar_points_plot.update_layout(font = dict(size = 15), 
                                               legend_font = dict(size = 14))
            similiar_points_plot.update_traces(diagonal_visible = False)
            similiar_points_plot.write_html('Pair Plot {:d}.html'.format(l + 1))
                  
    else:
        if pos_new != []:
            # If the entry in the data file is not present in the original Iris
            # dataset, concatenate it to the Iris dataframe
            df_iris = pd.concat(
                [df_iris, df_input.loc[[index for index in pos_new]]], 
                ignore_index = True)
            X_update = df_iris[
                ['Sepal Length/cm', 'Sepal Width/cm', 'Petal Length/cm', 
                 'Petal Width/cm']]
            X_update = StandardScaler().fit_transform(X_update)
            species_prediction = clf.predict(X)[
                [index for index in reversed(range(len(pos_new)))]]
            print('Species (Predicted):', ', '.join(species_prediction))
            for j in reversed(range(len(pos_new))):
                df_iris['Species'].loc[(len(df_iris.index)-1)-j] = species_prediction[j]
            nearest_neighbors = clf.kneighbors(X_update)[1]
            locations = []
            similiar_points = [[None for m in range(10)] for n in range(len(pos_new))]
            for k in reversed(range(len(pos_new))):
                locations.append(nearest_neighbors[(len(df_iris.index)-1)-k])
                similiar_points[k] = df_iris.loc[locations[k-1]]
            
            for l in reversed(range(len(similiar_points))):
                combined_df = pd.concat(
                    [df_iris.loc[[(len(df_iris.index)-1)-l]], similiar_points[l]],
                    ignore_index = True)
                similiar_points_plot = px.scatter_matrix(
                    combined_df, dimensions = ['Sepal Length/cm', 'Sepal Width/cm', 
                                               'Petal Length/cm', 'Petal Width/cm'],
                    color = combined_df['Species'], opacity = 1)
                similiar_points_plot.update_layout(font = dict(size = 15), 
                                                   legend_font = dict(size = 14))
                similiar_points_plot.update_traces(diagonal_visible = False)
                similiar_points_plot.write_html('Pair Plot {:d}.html'.format(l + 1))
            
        if pos_exist != []:
            # Predict the species
            species = clf.predict(X)[pos_exist]
            print('Species: ' + ', '.join(species))
            nearest_neighbors = clf.kneighbors(X)[1]
            # Indices of the 10 nearest neighbors (similiar points)
            locations = []
            similiar_points = [[None for a in range(10)] for b in range(len(pos_exist))]
            for c, index in enumerate(pos_exist):
                locations.append(nearest_neighbors[index])
                similiar_points[c] = df_iris.loc[locations[c]]
        
            for d in range(len(similiar_points)):
                combined_df = pd.concat([df_iris.loc[[pos_exist[d]]], similiar_points[d]], 
                                     ignore_index = True)
                similiar_points_plot = px.scatter_matrix(
                    combined_df, dimensions = ['Sepal Length/cm', 'Sepal Width/cm', 
                                               'Petal Length/cm', 'Petal Width/cm'],
                    color = combined_df['Species'], opacity = 1)
                similiar_points_plot.update_layout(font = dict(size = 15), 
                                                   legend_font = dict(size = 14))
                similiar_points_plot.update_traces(diagonal_visible = False)
                similiar_points_plot.write_html(
                    'Pair Plot {:d}.html'.format(d + len(pos_new) + 1))
        
if __name__ == '__main__':
    main(**vars(parse_args()))