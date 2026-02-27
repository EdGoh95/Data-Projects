#!/usr/bin/env python3
"""
Building Machine Learning Powered Applications Chapter 9: Choose Your Deployment Options
(For Streaming API Deployments)
"""
import pandas as pd
import joblib
import spacy
import json
from flask import *
from functools import lru_cache
from helper_functions import get_split_by_author, get_model_score_from_input, get_features_from_input, \
    get_recommendations_and_predictions_text
from scipy.sparse import hstack
from lime.lime_tabular import LimeTabularExplainer
initial_pipeline = __import__('Chapter 3 - Build Your First End-To-End Pipeline')

ML_Editor_App = Flask(__name__, template_folder = 'Source Code From GitHub/templates/')

posts_features_df = pd.read_csv('Data Science Posts With Features.csv')
questions_df = posts_features_df[posts_features_df['is_question']]
train_df, test_df = get_split_by_author(questions_df.copy(), test_size = 0.2)

with open('Features Mapping.json') as json_file:
    features_mapping = json.load(json_file)
feature_columns = list(features_mapping.keys())

@ML_Editor_App.route('/')
def landing_page():
    '''
    Renders the landing page template into a HTML webpage
    '''
    return render_template('landing.html')

@ML_Editor_App.route('/v1', methods = ['GET', 'POST'])
def v1():
    '''
    Renders the input form and results for the initial pipeline into a HTML webpage
    '''
    return handle_text_request(request, 'v1.html')

@ML_Editor_App.route('/v2', methods = ['GET', 'POST'])
def v2():
    '''
    Renders the input form and results for the first ML-powered pipeline (Model_v2) into a
    HTML webpage
    '''
    return handle_text_request(request, 'v2.html')

@ML_Editor_App.route('/v3', methods = ['GET', 'POST'])
def v3():
    '''
    Renders the input form and results for the improved ML-powered pipeline
    (Model_v3 - improved model with actionable recommendations/suggestions) into a HTML webpage
    '''
    return handle_text_request(request, 'v3.html')

@lru_cache(maxsize = 128)
def retrieve_recommendations(input_text, model):
    '''
    Computes and retrieves the recommendation for the requested pipeline/model.
    lru_cache stores the results of previous requests. If the same question has been submitted again,
    the cached results can be retrieved so as to speed up processing
    params:
        input_text - Input string
        model - Requested pipeline/model
    returns:
        Recommendations for the given question for the requested pipeline/model
    '''
    if model == 'v1':
        return initial_pipeline.get_recommendations_from_input(input_text)
    if model == 'v2':
        vectoriser = joblib.load('Models/Vectoriser_v2.pkl')
        classifier = joblib.load('Models/Model_v2.pkl')
        spacy_model = spacy.load('en_core_web_sm')

        vectors = vectoriser.transform([input_text])
        features = get_features_from_input(input_text, spacy_model, feature_columns).fillna(0).iloc[0]
        positive_probabilities = classifier.predict_proba(hstack([vectors, features]))[0][1]
        output_string = '''
        <b>Score</b> (Higher is better - 0 being the worst, 1 being the best): {:.2f}'''.format(
        positive_probabilities)
        return output_string
    if model == 'v3':
        classifier = joblib.load('Models/Model_v3.pkl')
        spacy_model = spacy.load('en_core_web_sm')
        LIME_explainer = LimeTabularExplainer(
            train_df[feature_columns].values, feature_names = feature_columns, class_names = ['Low', 'High'])

        score = get_model_score_from_input(input_text, feature_columns, spacy_model, classifier)
        features = get_features_from_input(input_text, spacy_model, feature_columns).fillna(0).iloc[0]
        explanation = LIME_explainer.explain_instance(features, classifier.predict_proba,
                                                      num_features = 10, labels = (1,))
        return get_recommendations_and_predictions_text(score, features_mapping, explanation)
    raise ValueError('Incorrect model selection!')

def handle_text_request(request, template_name):
    '''
    Renders the input form to receive GET requests and displays the results for the given question
    in the form of a POST request
    params:
        request - HTTP request
        template_name - Name of the template for the requested pipeline/model (v1, v2, v3)
    returns:
        Rendered input and results HTML webpage depending on the request type
    '''
    if request.method == 'POST':
        input_text = request.form.get('question')
        model = template_name.split('.')[0]
        suggestions = retrieve_recommendations(input_text, model)
        payload = {'input': input_text, 'suggestions': suggestions, 'model_name': model}
        return render_template('results.html', ml_result = payload)
    else:
        return render_template(template_name)

if __name__ == '__main__':
    ML_Editor_App.run()