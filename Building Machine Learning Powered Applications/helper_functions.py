#!/usr/bin/env python3
"""
Building Machine Learning Powered Applications
"""
import xml.etree.ElementTree as ELT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import nltk
# nltk.download('vader_lexicon')
from tqdm import tqdm
tqdm.pandas()
from bs4 import BeautifulSoup
from keras.preprocessing import image
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix, roc_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve
from scipy.sparse import vstack, hstack
from nltk.sentiment.vader import SentimentIntensityAnalyzer

POS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN',
       'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']


def parse_xml_to_csv(path, save_path = None):
    '''
    Opens the .xml posts, convert and dump the text into a csv file, tokenizing it in the process
    params:
        path - Path to the XML file containing the posts
    returns:
        A dataframe of the processed text
    '''
    # Parsing the XML file
    doc = ELT.parse(path)
    root = doc.getroot()

    all_rows = [row.attrib for row in root.findall('row')] # Each row being a question
    # Since preprocessing requires time, progress is displayed with the help of tqdm
    for item in tqdm(all_rows):
        soup = BeautifulSoup(item['Body'], features = 'html.parser') # Decoding text from HTML
        item['body_text'] = soup.get_text()

    df = pd.DataFrame.from_dict(all_rows)
    if save_path:
        df.to_csv(save_path, index = False)
    return df

def generate_image_features(image_paths):
    '''
    Takes in an array of image paths and then returns the features for each image obtained from the
    pre-trained model
    params:
        image_paths - An array of the image paths
    return:
        An array of the last-layer activations and mappings from array_index to filepath
    '''
    images = np.zeros(shape = (len(image_paths), 224, 224, 3))
    # Load the pre-trained VGG16 model
    pretrained_VGG16 = VGG16(weights = 'imagenet', include_top = True)
    # Using only the penultimate layer to leverage the learnt features
    model = Model(inputs = pretrained_VGG16.input, outputs = pretrained_VGG16.get_layer('fc2').output)
    for i, f in enumerate(image_paths):
        img = image.load_img(f, target_size = (224, 224))
        x_raw = image.img_to_array(img)
        x_expand = np.expand_dims(x_raw, axis = 0)
        images[i, :, :, :] = x_expand

    # Loading all the images and passing them to the pre-trained VGG16 model
    inputs = preprocess_input(images)
    images_features = model.predict(inputs)
    return images_features

def plot_embeddings(embeddings, sentence_labels):
    '''
    Plots the embeddings, coloured based on the sentence label
    params:
        embeddings: 2-Dimensional (2D) embeddings
        sentence_labels: Labels to be displayed
    '''
    plt.figure(figsize = (16, 10))
    colour_map = {True: 'tab:orange', False: 'tab:blue'}
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c = [colour_map[x] for x in sentence_labels],
                s = 40, alpha = 0.4)
    handles = [Rectangle((0, 0), 1, 1, color = c, ec = 'k') for c in ['tab:orange', 'tab:blue']]
    labels = ['Answered', 'Unanswered']
    plt.legend(handles, labels)
    plt.gca().set_aspect('equal', 'box')
    plt.gca().set_xlabel('x')
    plt.gca().set_ylabel('y')

def get_normalised_series(df, col):
    '''
    Returns the normalised version of a column in the dataframe
    params:
        df - Input dataframe
        col - Column to be normalised
    returns:
        Series normalised using the standard score (Z-score)
    '''
    return (df[col] - df[col].mean())/df[col].std()

def format_raw_df(posts_df):
    '''
    Cleans the raw dataframe
    params:
        posts_df - Raw dataframe
    returns:
        Processed dataframe
    '''
    count_columns = posts_df.filter(regex = 'Count').columns
    posts_df[count_columns] = posts_df[count_columns].fillna(-1).astype(int)
    id_columns = posts_df.filter(regex = 'Id').columns
    posts_df[id_columns] = posts_df[id_columns].fillna(-1).astype(int)
    posts_df.set_index('Id', inplace = True, drop = False)

    posts_df['is_question'] = posts_df['PostTypeId'] == 1
    posts_df = posts_df[posts_df['PostTypeId'].isin([1, 2])]
    return posts_df

def get_randomised_train_test_split(posts_df, test_size = 0.3, random_state = 40):
    '''
    Splits the dataframe into train and test sets. Assumes the dataframe only has one row per
    question example
    params:
        posts_df - Dataframe containing all the posts, with their labels
        test_size - Proportion of the dataframe allocated to the test set
        random_state - A random seed for initialisation
    '''
    return train_test_split(posts_df, test_size = test_size, random_state = random_state)

def get_split_by_author(posts_df, author_id_column = 'OwnerUserId', test_size = 0.3, random_state = 40):
    '''
    Splits the dataframe into train and test sets while ensuring that every author only appears in
    one of the splits
    params:
        posts_df - Dataframe containing all the posts, with their labels
        author_id_column - Name of the column containing the author_id
        test_size - Proportion of the dataframe allocated to the test set
        random_state - A random seed for initialisation
    '''
    group_splitter = GroupShuffleSplit(n_splits = 1, test_size = test_size, random_state = random_state)
    group_splits = group_splitter.split(posts_df, groups = posts_df[author_id_column])
    train_index, test_index = next(group_splits)
    return posts_df.iloc[train_index, :], posts_df.iloc[test_index, :]

def get_vectorised_series(text_series, vectoriser):
    '''
    Vectorises the input text series using the vectoriser pre-trained on the training set
    params:
        text_series - Series containing the text to be vectorised
        vectoriser - Vectoriser that has fitted on the training set
    returns:
        An array containing the vectorised features
    '''
    vectors = vectoriser.transform(text_series)
    return [vectors[i] for i in range(vectors.shape[0])]

def get_feature_vectors_and_labels(df, feature_columns):
    '''
    Generates the feature vectors and the corresponding labels
    params:
        df - Input train/test dataframe
        feature_columns - Columns containing the generated features
    returns:
        Features array and labels array
    '''
    vectorised_features = vstack(df['vectors'])
    added_features = df[feature_columns].astype(float)
    features = hstack([vectorised_features, added_features])
    labels = df['Score'] > df['Score'].median()
    return features, labels

def get_average_word_length(tokens):
    '''
    Returns the average word length for a list of words
    params:
        tokens - List of words
    returns:
        Average number of characters per word
    '''
    if len(tokens) < 1:
        return 0
    token_lengths = [len(token) for token in tokens]
    return float(sum(token_lengths)/len(token_lengths))

def count_POS_occurrences(df):
    '''
    Counts the number of occurrences for each Part of Speech (POS) and adds them to the input dataframe
    params:
        df - Dataframe containing text that has been passed to the NLP_model (usually a spaCy model)
    returns:
        Dataframe containing the number of occurrences for each POS
    '''
    POS_list = df['spacy_text'].apply(lambda doc: [token.pos_ for token in doc])
    for pos in POS:
        df[pos] = POS_list.apply(lambda x: len([tag for tag in x if tag == pos]))/df['text_length']
    return df

def get_polarity_score(questions_df):
    '''
    Calculates the polarity score for each question using a function from the nltk package and adds
    them, to the input dataframe
    params:
        questions_df - Dataframe containing questions
    returns:
        Dataframe containing the polarity scores
    '''
    SID = SentimentIntensityAnalyzer()
    questions_df['polarity'] = questions_df['full_text'].progress_apply(lambda x: SID.polarity_scores(x)['pos'])
    return questions_df

def add_text_features(questions_df):
    '''
    Adds text features to the input dataframe
    params:
        questions_df - Dataframe containing questions
    returns:
        Dataframe with the added text features
    '''
    questions_df['question_mark_full'] = questions_df['full_text'].str.contains('?', regex = False)

    questions_df['question_word_full'] = (
        questions_df['full_text'].str.contains('how', regex = False, case = False) |
        questions_df['full_text'].str.contains('what', regex = False, case = False) |
        questions_df['full_text'].str.contains('where', regex = False, case = False))
    questions_df['action_verb_full'] = (
        questions_df['full_text'].str.contains('can', regex = False, case = False) |
        questions_df['full_text'].str.contains('will', regex = False, case = False))

    questions_df['text_length'] = questions_df['full_text'].str.len()
    questions_df['normalised_text_length'] = get_normalised_series(questions_df, 'text_length')
    return questions_df

def add_text_stats_features(questions_df, NLP_model):
    '''
    Adds statistical features, such as counts of punctuation marks, characters and words, to the
    input dataframe
    params:
        questions_df - Dataframe containing questions
    returns:
        Dataframe with the added statistical features
    '''
    questions_df['spacy_text'] = questions_df['full_text'].progress_apply(lambda x: NLP_model(x))
    questions_df['num_words'] = 100 * (questions_df['spacy_text'].apply(lambda x: len(x)) /
                                       questions_df['text_length'])
    questions_df['num_unique_words'] = questions_df['spacy_text'].apply(lambda x: len(set(x)))
    questions_df['num_stop_words'] = 100 * (
        questions_df['spacy_text'].apply(lambda x: len([word for word in x if word.is_stop])) /
        questions_df['text_length'])
    questions_df['average_word_length'] = questions_df['spacy_text'].apply(
        lambda x: get_average_word_length(x))

    questions_df['num_question_marks'] = 100 * (questions_df['full_text'].str.count('\?') /
                                           questions_df['text_length'])
    questions_df['num_full_stops'] = 100 * (questions_df['full_text'].str.count('\.') /
                                         questions_df['text_length'])
    questions_df['num_commas'] = 100 * (questions_df['full_text'].str.count(',') /
                                        questions_df['text_length'])
    questions_df['num_exclamation_marks'] = 100 * (questions_df['full_text'].str.count('!') /
                                                   questions_df['text_length'])
    questions_df['num_colons'] = 100 * (questions_df['full_text'].str.count(':') /
                                        questions_df['text_length'])
    questions_df['num_semicolons'] = 100 * (questions_df['full_text'].str.count(';') /
                                            questions_df['text_length'])
    questions_df['num_quotes'] = 100 * (questions_df['full_text'].str.count('"') /
                                                  questions_df['text_length'])
    return questions_df

def add_features(posts_df, NLP_model):
    '''
    Adds features to the dataframe
    params:
        posts_df - Dataframe containing all the posts, with their labels
    returns:
        Dataframe with the generated features
    '''
    posts_df['full_text'] = posts_df['Title'].str.cat(posts_df['body_text'], sep = ' ', na_rep = '')
    posts_df = add_text_features(posts_df.copy())
    posts_df = add_text_stats_features(posts_df.copy(), NLP_model)
    posts_df = count_POS_occurrences(posts_df.copy())
    posts_df = get_polarity_score(posts_df.copy())
    return posts_df

def get_metrics(ground_truth, prediction):
    '''
    Gets the standard metrics for binary classification
    params:
        ground_truth - Actual labels
        prediction - Labels predicted by the model
    returns:
        The accuracy, precision, recall and F1 scores of the model
    '''
    accuracy = accuracy_score(ground_truth, prediction)
    precision = precision_score(ground_truth, prediction, pos_label = None, average = 'weighted')
    recall = recall_score(ground_truth, prediction, pos_label = None, average = 'weighted')
    F1 = f1_score(ground_truth, prediction, pos_label = None, average = 'weighted')
    return accuracy, precision, recall, F1

def plot_confusion_matrix(ground_truth, prediction, classes = None, normalise = False,
                          title = 'Confusion Matrix', cmap = plt.get_cmap('binary'), figsize = (16, 10)):
    '''
    Inspired by https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    params:
        ground_truth - Actual labels
        prediction - Labels predicted by the model
        classes - Names of the classes
        normalise - Whether the plot should be normalised
        title - Plot title
        cmap - Which colormap to use
        figsize - Size of the output plot
    returns:
        Plot of the confusion matrix
    '''
    if classes is None:
        classes = ['Low Quality', 'High Quality']
    cmatrix = confusion_matrix(ground_truth, prediction)
    if normalise:
        cmatrix = cmatrix.astype(float)/cmatrix.sum(axis = 1)[:, np.newaxis]

    plt.figure(figsize = figsize)
    axe = plt.gca()
    image = axe.imshow(cmatrix, interpolation = 'nearest', cmap = cmap)
    plt.colorbar(image)

    fmt = '.2f' if normalise else 'd'
    threshold = ((cmatrix.max() - cmatrix.min())/2.0) + cmatrix.min()
    for i, j in itertools.product(range(cmatrix.shape[0]), range(cmatrix.shape[1])):
        plt.text(j, i, format(cmatrix[i, j], fmt), horizontalalignment = 'center',
                 color = 'white' if cmatrix[i, j] > threshold else 'black', fontsize = 40)

    title = plt.title(title, fontsize = 25)
    title.set_position([0.5, 1.15])
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize = 15)
    plt.xlabel('Predicted Label', fontsize = 20)
    plt.yticks(tick_marks, classes, fontsize = 15)
    plt.ylabel('Actual Label', fontsize = 20)
    plt.tight_layout()

def plot_ROC_curve(ground_truth, predicted_probabilities, TPR_line = -1, FPR_line = -1,
                   figsize = (16, 10)):
    '''
    Inspired by https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    params:
        ground_truth - Actual labels
        predicted_probabilities - Predicted probabilities of the model for each example
        TPR_line - A line representing a threshold True Positive Rate (TPR_ value
        FPR_line - A line representing a threshold False Positive Rate (FPR) value
        figsize - Size of the output plot
    returns:
        Plot of the ROC curve
    '''
    FPR, TPR, thresholds = roc_curve(ground_truth, predicted_probabilities)
    ROC_AUC = auc(FPR, TPR)

    plt.figure(figsize = figsize)
    plt.plot(FPR, TPR, lw = 1, alpha = 1, color = 'black',
             label = 'ROC Curve (AUC = {:.3f}'.format(ROC_AUC))
    plt.plot([0, 1], [0, 1], linestyle = '--', lw = 2, alpha = 1, color = 'grey', label = 'Chance')
    plt.plot([0.01, 0.01, 1], [0.01, 0.99, 0.99], linestyle = ':', lw = 2, alpha = 1, color = 'green',
             label = 'Perfect Model')

    if TPR_line != -1:
        plt.plot([0, 1], [TPR_line, TPR_line], linestyle = '-', lw = 2, alpha = 1, color = 'red',
                 label = 'TPR Requirement')
        plt.fill_between([0, 1], [TPR_line, TPR_line], [1, 1], alpha = 0, hatch = '\\')
    if FPR_line != -1:
        plt.plot([FPR_line, FPR_line], [0, 1], linestyle = '-', lw = 2, alpha = 1, color = 'red',
                 label = 'FPR Requirement')
        plt.fill_between([FPR_line, 1], [1, 1], alpha = 0, hatch = '\\')

    plt.legend(loc = 'lower right')
    plt.xlabel('False Positive Rate (FPR)', fontsize = 20)
    plt.xlim(0, 1)
    plt.ylabel('True Positive Rate (TPR)', fontsize = 20)
    plt.ylim(0, 1)

def plot_calibration_curve(ground_truth, predicted_probabilities, figsize = (16, 10)):
    '''
    Inspired by https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html
    params:
        ground_truth - Actual labels
        predicted_probabilities - Predicted probabilities of the model for each example
        figsize - Size of the output plot
    returns:
        Plot of the calibration curve
    '''
    plt.figure(figsize = figsize)
    axe1 = plt.subplot2grid((3, 1), (0, 0), rowspan = 2)
    axe2 = plt.subplot2grid((3, 1), (2, 0))

    classifier_score = brier_score_loss(ground_truth, predicted_probabilities,
                                        pos_label = ground_truth.max())
    print('\tBrier Score = {:1.3f}'.format(classifier_score))

    fraction_positives, mean_predicted_value = calibration_curve(ground_truth, predicted_probabilities,
                                                                  n_bins = 10)
    axe1.plot([0, 1], [0, 1], 'k:', label = 'Perfectly Calibrated')
    axe1.plot(mean_predicted_value, fraction_positives, 's-', color = 'black',
              label = 'Brier Score = {:1.3f} (0 being the best and 1 being the worst)'.format(classifier_score))
    axe1.set_title('Calibration Plot')
    axe1.set_xlim([0, 1])
    axe1.set_ylim([0, 1])
    axe1.set_ylabel('Fraction of Positives')
    axe1.legend(loc = 'lower right')

    axe2.hist(predicted_probabilities, range = (0, 1), bins = 10, histtype = 'step', lw = 2,
              color = 'black')
    axe2.set_title('Probability Distribution')
    axe2.set_xlabel('Mean Predicted Value')
    axe2.set_ylabel('Count')
    axe2.legend(loc = 'upper center', ncol = 2)

    plt.tight_layout()

def plot_multiple_calibration_curves(ground_truth_array, predicted_probabilities_array, figsize = (16, 10)):
    '''
    Inspired by https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html
    params:
        ground_truth_array - List of actual labels for each model
        predicted_probabilities_array - Matrix containing the predicted probabilities of each model
                                        for each example
        figsize - Size of the output plot
    returns:
        Multiple calibration curves for the comparison of multiple models within one plot
    '''
    plt.figure(figsize = figsize)
    axe1 = plt.subplot2grid((3, 1), (0, 0), rowspan = 2)
    axe2 = plt.subplot2grid((3, 1), (2, 0))

    axe1.plot([0, 1], [0, 1], 'k:', label = 'Perfectly Calibrated')

    for index, (ground_truth, predicted_probabilities) in enumerate(zip(
            ground_truth_array, predicted_probabilities_array)):
        classifier_score = brier_score_loss(ground_truth, predicted_probabilities,
                                            pos_label = ground_truth.max())

        fraction_positives, mean_predicted_value = calibration_curve(
            ground_truth, predicted_probabilities, n_bins = 10)
        axe1.plot(
            mean_predicted_value, fraction_positives, 's-',
            label = 'Model {:d}: Brier Score = {:1.3f} (0 being the best and 1 being the worst)'.format(
                      (index + 1), classifier_score))
        axe1.set_title('Calibration Plot')
        axe1.set_xlim([0, 1])
        axe1.set_ylim([0, 1])
        axe1.set_ylabel('Fraction of Positives')
        axe1.legend(loc = 'lower right')

        axe2.hist(predicted_probabilities, range = (0, 1), bins = 10, histtype = 'step', lw = 2,
                  label = 'Model {:d}'.format(index + 1))
        axe2.set_title('Probability Distribution')
        axe2.set_xlabel('Mean Predicted Value')
        axe2.set_ylabel('Count')
        axe2.legend(loc = 'upper right', ncol = 2)

    plt.tight_layout()

def get_top_k(evaluation_df, ground_truth, predicted_probabilities, k = 5, decision_threshold = 0.5):
    '''
    For evaluation of binary classification models
    Returns the k most correct and incorrect examples for each class. Also returns the k most unsure
    examples
    params:
        evaluation_df - Dataframe containing the predictions and ground truths
        ground_truth - Column containing the actual labels
        predicted_probabilities - Column containing the predicted probabililties
        k - Number of examples to show for each category
        decision_threshold - Decision boundary of the classifier, above which examples will be
                             classified as positive
    returns:
        correct_positive, correct_negative, incorrect_positive, incorrect_negative, most_unsure
    '''
    correct_predictions = evaluation_df[
        (evaluation_df[predicted_probabilities] > decision_threshold) == evaluation_df[ground_truth]].copy()
    top_correct_posiitive = correct_predictions[correct_predictions[ground_truth]].nlargest(
        k, predicted_probabilities)
    top_correct_negative = correct_predictions[~correct_predictions[ground_truth]].nsmallest(
        k, predicted_probabilities)

    incorrect_predictions = evaluation_df[
        (evaluation_df[predicted_probabilities] > decision_threshold) != evaluation_df[ground_truth]].copy()
    top_incorrect_positive = incorrect_predictions[incorrect_predictions[ground_truth]].nsmallest(
        k, predicted_probabilities)
    top_incorrect_negative = incorrect_predictions[~incorrect_predictions[ground_truth]].nlargest(
        k, predicted_probabilities)

    # Get the closest examples to the decision threshold
    most_uncertain = evaluation_df.iloc[
        (evaluation_df[predicted_probabilities] - decision_threshold).abs().argsort()[:k]]
    return (top_correct_posiitive, top_correct_negative,
            top_incorrect_positive, top_incorrect_negative, most_uncertain)

def get_feature_importance(classifier, features):
    '''
    Gets the list of feature importances for a given classifier
    params:
        classifier - Any scikit-learn classifier
        features - List of features in the order they were passed to the classifier
    returns:
        A sorted list of tuples of the form (features, importance_score)
    '''
    importances = classifier.feature_importances_
    sorted_by_importance = np.argsort(importances)[::-1]
    return list(zip(features[sorted_by_importance], importances[sorted_by_importance]))

def get_features_from_input(input_text, NLP_model, feature_columns):
    '''
    Generates features for the input text
    params:
        input_text - Input string
        feature_columns - Columns containing the generated features
    returns:
        Dataframe containing the features
    '''
    text_df = pd.DataFrame([input_text], columns = ['full_text'])
    text_df = add_text_features(text_df.copy())
    text_df = add_text_stats_features(text_df.copy(), NLP_model)
    text_df = count_POS_occurrences(text_df.copy())
    text_df = get_polarity_score(text_df.copy())
    features_df = text_df[feature_columns].astype(float)
    return features_df

def get_model_score_from_input(input_text, feature_columns, NLP_model, classifier):
    '''
    Gets the model score in terms of probability for the given input_text
    params:
        input_text - Input string
        feature_columns - Columns containing the generated features
        classifier - Any pre-trained scikit-learn classifier
    returns:
        Estimated probability of a given question receiving a high score
    '''
    features = get_features_from_input(input_text, NLP_model, feature_columns)

    probabilities = classifier.predict_proba(features)
    positive_probabilities = probabilities[0][1]
    return positive_probabilities

def simplify_comparison(comparison):
    '''
    Simplifies the comparison for clearer display of recommendations to users
    params:
        comparison - Input comparison operator
    returns:
        Simplified comparison operator
    '''
    if comparison in ['<=', '<']:
        return '<'
    if comparison in ['>=', '>']:
        return '>'
    return comparison

def get_recommended_modification(comparison, contribution):
    '''
    Gets the recommended modification based on the direction of the comparison operator and the sign
    of the contribution
    params:
        comparison - Simplified comparison operator
        contribution - Contribution, whether it is positive or negative
    returns:
        Recommended modification to improve the quality of the question
    '''
    if comparison == '<' and contribution < 0:
        return 'Increase'
    if comparison == '>' and contribution < 0:
        return 'Reduce'
    if comparison == '<' and contribution > 0:
        return 'No need to increase'
    if comparison == '>' and contribution > 0:
        return 'No need to reduce'

def parse_explanations(explanation_list, features_mapping):
    '''
    Transforms the explanations returned by the LIME black box explainer into a user-friendly format
    params:
        explanation_list - Explanations returned by the LIME explainer (as a list)
        features_mapping - Dictionary containing the original and display names of the generated features
    returns:
        Dataframe containing the explanations in a user-friendly format
    '''
    parsed_explanations = pd.DataFrame(
        columns = ['Feature (Original)', 'Feature (Display Name)', 'Direction', 'Threshold',
                   'Contribution', 'Recommendation'])
    for index, (feature_value, contribution) in enumerate(explanation_list):
        conditions = feature_value.split(' ')
        # Ignoring doubly bounded conditions (between) which are harder to formulate as a recommendation
        if len(conditions) == 3:
            feature_name, comparison, threshold =  conditions

            simplified_comparison = simplify_comparison(comparison)
            recommended_modification = get_recommended_modification(simplified_comparison, contribution)

            parsed_explanations.loc[index] = [feature_name, features_mapping[feature_name],
                                              simplified_comparison, threshold, contribution,
                                              recommended_modification]
    return parsed_explanations

def get_recommendations_from_parsed_explanations(parsed_explanations):
    '''
    Generates clear recommendations to be displayed to users
    params:
        parsed_explanations - Dataframe containing the LIME explanations
    returns:
        A list of recommendations to improve the quality of the given question
    '''
    recommendations = []
    for index, suggested_modification in enumerate(parsed_explanations.to_dict(orient = 'records')):
        recommendation = '{} {}'.format(suggested_modification['Recommendation'],
                                        suggested_modification['Feature (Display Name)'])
        font_color = 'green'
        if suggested_modification['Recommendation'] in ['Increase', 'Reduce']:
            font_color = 'red'
        recommendations.append('<font color="{}"> {}) {}</font>'.format(font_color, index + 1,
                                                                        recommendation))
    recommendations_string = '<br/>'.join(recommendations)
    return recommendations_string

def get_recommendations_and_predictions_text(model_score, features_mapping, explanation):
    '''
    Evaluates the quality and displays recommendations based on the given question
    params:
        model_score - Score of the given question obtained from the get_model_score_from_input function
        features_mapping - Dictionary containing the original and display names of the generated features
        explanation - Explanation obtained from the LIME black box explainer
    returns:
        Model score (quality) together with recommendations for the given question in HTML format
    '''
    parsed_explanations = parse_explanations(explanation.as_list(), features_mapping)
    recommendations = get_recommendations_from_parsed_explanations(parsed_explanations)
    output_string = '''<b>Score</b> (Higher is better - 0 being the worst, 1 being the best): {:.2f}
                       <br/>
                       <b>Recommendations</b> (in descending order of importance):
                       <br/> {}'''.format(model_score, recommendations)
    return output_string