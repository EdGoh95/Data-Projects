#!/usr/bin/env python3
"""
Machine Learning Infrastructure And Best Practices For Software Engineers (Packt Publishing)
Chapter 12: Designing Machine Learning Pipelines (MLOps) and Their Testing
"""
import json
import pandas as pd
import joblib
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

#%% Feature-Based ML Pipelines
BERT_tokenizer = AutoTokenizer.from_pretrained('mstaron/SingBERTa')
BERT_model = AutoModelForMaskedLM.from_pretrained('mstaron/SingBERTa')

BERT_extracted_features = pipeline('feature-extraction', model = BERT_model, tokenizer = BERT_tokenizer,
                                   return_tensor = False)

#### Feature Extraction Pipeline Testing
def test_features():
    # Get the embeddings of the word 'Test'
    features_list = BERT_extracted_features('Test')
    with open('../Source Code From GitHub/chapter_12/test.json', 'r') as json_file:
        embeddings_list = json.load(json_file)

    assert features_list[0][0] == embeddings_list

#### Zero-Table Test
decision_tree_model = joblib.load('../Source Code From GitHub/chapter_12/chapter_12_decision_tree_model.joblib')
ant_df = pd.read_excel(
    "../Chapter 6 - Processing Data In Machine Learning Systems/Regenerated PROMISE and BPD Datasets.xlsx",
    sheet_name = 'ant_1_3', index_col = 0)
ant_features = ant_df.drop(['Defect'], axis = 1)

def test_model_not_null():
    assert decision_tree_model is not None

def test_model_predicts_yes_correctly():
    assert decision_tree_model.predict(ant_features)[0] == 1

def test_model_predicts_class_no_correctly():
    assert decision_tree_model.predict(ant_features)[1] == 0