#!/usr/bin/env python3
"""
Hands-On Ensemble Learning with Python (Packt Publishing) Chapter 9:
Classifying Fraudulent Transactions
"""
import time
import sys
sys.path.append(
    '../../../../Hands-On Ensemble Learning with Python (Master)/Chapter09/')
import numpy as np 
import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)
import matplotlib.pyplot as plt
from termcolor import colored
colours = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'grey', 'white']
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from stacking_classifier import Stacking
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, recall_score

start = time.time()
#%% Load The Dataset
credit_card_df = pd.read_csv('Data/Credit Card Fraud Detection.csv')

#%% Data Exploration/Exploratory Analysis
Count = []
Mean, Std = [], []
Min, Max = [], []
for column in credit_card_df.columns[:-1]:
    Count.append(credit_card_df[column].count())
    Mean.append(credit_card_df[column].mean())
    Std.append(credit_card_df[column].std())
    Min.append(credit_card_df[column].min())
    Max.append(credit_card_df[column].max())
 
# Descriptive statistics of the dataset
summary_df = pd.DataFrame({'Count': Count, 'Mean': Mean, 'Std': Std, 
                           'Min': Min, 'Max': Max})
summary_df.index = credit_card_df.columns[:-1]
print('\u2500'*33 + ' Descriptive (Summary) Statistics ' + '\u2500'*33)
display(summary_df.T)

# Plot histogram for each feature/column
plt.figure(1)
for i, column in enumerate(credit_card_df.columns[:-1]):
    plt.subplot(5, 6, i+1)
    credit_card_df[column].hist(bins = 50, figsize = (15, 10))
    plt.title('{}'.format(column))
plt.subplots_adjust(left = None, bottom = None, right = None, top = None, 
                    wspace = 0.4, hspace = 0.8)
plt.tight_layout()

# Plot the 'Amount' in logarithmic scale
plt.figure(2)
credit_card_df['Amount'].plot.hist(bins = 50, logy = True, title = 'Amount')

#### Standardize The 'Time' and 'Amount' Columns
# This ensures an even distribution of weights between individual features
# Plot the standardized 'Amount' in logarithmic scale
credit_card_df['Amount'] = (
    credit_card_df['Amount'] - credit_card_df['Amount'].mean())/\
    credit_card_df['Amount'].std()
plt.figure(3)
credit_card_df['Amount'].plot.hist(bins = 50, logy = True, 
                                   title = 'Amount (Standardized)')

# Plot the standardized 'Time'
credit_card_df['Time'] = (
    credit_card_df['Time'] - credit_card_df['Time'].min())/credit_card_df['Time'].std()
plt.figure(4)
credit_card_df['Time'].plot.hist(bins = 50, title = 'Time (Standardized)')
#%% Split The Dataset Into Training (70%) And Testing (30%) Sets
np.random.seed(123456)
credit_card_train_features, credit_card_test_features, \
    credit_card_train_target, credit_card_test_target = \
        train_test_split(credit_card_df.drop('Class', axis = 1).values, 
                         credit_card_df['Class'].values, test_size = 0.3)
        
#%% Evaluating The Base Learners
base_learners = [
    ('Decision Tree Classifier', DecisionTreeClassifier(max_depth = 5)),
    ('Gaussian Naive Bayes Classifer', GaussianNB()), 
    ('Logistic Regression', LogisticRegression())]
print('\u2500'*29 + " Evaluation of Base Learner's Performance " + '\u2500'*29)
for base_learner, colour in zip(base_learners, colours):
    base_learner[1].fit(credit_card_train_features, credit_card_train_target)
    predictions_base = base_learner[1].predict(credit_card_test_features)
    print(colored(base_learner[0], colour, attrs = ['bold']))
    print('F1 score: {:.3f}'.format(f1_score(credit_card_test_target, 
                                             predictions_base)))
    print('Recall score: {:.3f}\n'.format(recall_score(credit_card_test_target, 
                                                       predictions_base)))

#### Plotting The Correlations
correlations_sorted = credit_card_df.corr()['Class'].drop('Class').sort_values()
plt.figure(5)
correlations_sorted.plot(kind = 'bar', title = 'Correlations to Class',
                         color = plt.cm.tab10(range(11)))

#### Filtering Out Features With Low Absolute Correlations To The Target
filtered = list(correlations_sorted[np.abs(correlations_sorted) > 0.1].index.values)
filtered.append('Class')
credit_card_filtered_df = credit_card_df[filtered]

# Split the filtered dataset into training (70%) and testing (30%) Sets
np.random.seed(123456)
credit_card_filtered_train_features, credit_card_filtered_test_features, \
    credit_card_filtered_train_target, credit_card_filtered_test_target = \
        train_test_split(credit_card_filtered_df.drop('Class', axis = 1).values,
                         credit_card_filtered_df['Class'].values, test_size = 0.3)

print(colored('After Filtering:', 'grey', attrs = ['bold']))
for base_learner, colour in zip(base_learners, colours):
    base_learner[1].fit(credit_card_filtered_train_features, 
                        credit_card_filtered_train_target)
    predictions_base_filtered = base_learner[1].predict(
        credit_card_filtered_test_features)
    print(colored(base_learner[0], colour, attrs = ['bold']))
    print('F1 score: {:.3f}'.format(f1_score(
        credit_card_filtered_test_target, predictions_base_filtered)))
    print('Recall score: {:.3f}\n'.format(
        recall_score(credit_card_filtered_test_target, predictions_base_filtered)))
    
#### Optimizing The Decision Tree Classifier
depths = [j for j in range(3, 12)]
f1, recall = [], []
f1_filtered, recall_filtered = [], []
for depth in depths:
    DTC_optimized = DecisionTreeClassifier(max_depth = depth)
    # Original data
    DTC_optimized.fit(credit_card_train_features, credit_card_train_target)
    f1.append(f1_score(credit_card_test_target, 
                       DTC_optimized.predict(credit_card_test_features)))
    recall.append(recall_score(credit_card_test_target,
                               DTC_optimized.predict(credit_card_test_features)))
    
    # Filtered data
    DTC_optimized.fit(credit_card_filtered_train_features,
                      credit_card_filtered_train_target)
    f1_filtered.append(
        f1_score(credit_card_filtered_test_target, DTC_optimized.predict(
            credit_card_filtered_test_features)))
    recall_filtered.append(
        recall_score(credit_card_filtered_test_target, DTC_optimized.predict(
            credit_card_filtered_test_features)))

plt.figure(6)
plt.plot(depths, f1, label = 'F1 (Original)')
plt.plot(depths, recall, label = 'Recall (Original)')
plt.plot(depths, f1_filtered, label = 'F1 (Filtered)')
plt.plot(depths, recall_filtered, label = 'Recall (Filtered)')
plt.xlabel('Max Depth')
plt.ylabel('Score')
plt.legend(loc = 'best')

#%% Ensemble Methods
# Instantiate an array to store the F1 scores
f1_scores = []
recall_scores = []
#### Voting
voting_ensemble = VotingClassifier(base_learners)
# Original data
voting_ensemble.fit(credit_card_train_features, credit_card_train_target)
print('\u2500'*23 + ' Performance Evaluation of Different Ensemble Methods ' + 
      '\u2500'*23)
print(colored('Voting', 'yellow', attrs = ['bold']))
print('F1 score: {:.3f}'.format(
    f1_score(credit_card_test_target, voting_ensemble.predict(
        credit_card_test_features))))
print('Recall score: {:.3f}'.format(
    recall_score(credit_card_test_target, voting_ensemble.predict(
        credit_card_test_features))))

# Filtered data
voting_ensemble.fit(credit_card_filtered_train_features, 
                    credit_card_filtered_train_target)
print('F1 score (filtered): {:.3f}'.format(
    f1_score(credit_card_filtered_test_target, voting_ensemble.predict(
        credit_card_filtered_test_features))))
print('Recall score (filtered): {:.3f}'.format(
    recall_score(credit_card_filtered_test_target, voting_ensemble.predict(
        credit_card_filtered_test_features))))

# Diversify the voting ensemble to improve its performance
base_learners_diversified = base_learners.copy()
base_learners_diversified.extend([
    ('Decision Tree Classifier 2', DecisionTreeClassifier(max_depth = 3)),
    ('Decision Tree Classifier 3', DecisionTreeClassifier(max_depth = 8))])
voting_ensemble_diversified = VotingClassifier(base_learners_diversified)
# Original data
voting_ensemble_diversified.fit(credit_card_train_features, credit_card_train_target)
print(colored('\nAfter Diversification:', 'grey', attrs = ['bold']))
print('F1 score: {:.3f}'.format(
    f1_score(credit_card_test_target, voting_ensemble_diversified.predict(
        credit_card_test_features))))
f1_scores.append(('Voting (Original)', f1_score(
    credit_card_test_target, voting_ensemble_diversified.predict(
        credit_card_test_features))))
print('Recall score: {:.3f}'.format(
    recall_score(credit_card_test_target, voting_ensemble_diversified.predict(
        credit_card_test_features))))
recall_scores.append(('Voting (Original)', recall_score(
    credit_card_test_target, voting_ensemble_diversified.predict(
        credit_card_test_features))))

# Filtered data
voting_ensemble_diversified.fit(credit_card_filtered_train_features, 
                                credit_card_filtered_train_target)
print('F1 score (filtered): {:.3f}'.format(f1_score(
    credit_card_filtered_test_target, voting_ensemble_diversified.predict(
        credit_card_filtered_test_features))))
f1_scores.append(('Voting (Filtered)', f1_score(
    credit_card_filtered_test_target, voting_ensemble_diversified.predict(
        credit_card_filtered_test_features))))
print('Recall score (filtered): {:.3f}'.format(recall_score(
    credit_card_filtered_test_target, voting_ensemble_diversified.predict(
        credit_card_filtered_test_features))))
recall_scores.append(('Voting (Filtered)', recall_score(
    credit_card_filtered_test_target, voting_ensemble_diversified.predict(
        credit_card_filtered_test_features))))

#### Stacking
stacking_ensemble = Stacking(learner_levels = [[
    base_learners[k][1] for k in range(len(base_learners))], 
    [LogisticRegression()]])
print(colored('\nStacking', 'magenta', attrs = ['bold']))
# Original data
stacking_ensemble.fit(credit_card_train_features, credit_card_train_target)
print('F1 score: {:.3f}'.format(
    f1_score(credit_card_test_target, stacking_ensemble.predict(
        credit_card_test_features))))
print('Recall score: {:.3f}'.format(
    recall_score(credit_card_test_target, stacking_ensemble.predict(
        credit_card_test_features))))

# Filtered data
stacking_ensemble.fit(credit_card_filtered_train_features, 
                      credit_card_filtered_train_target)
print('F1 score (filtered): {:.3f}'.format(
    f1_score(credit_card_filtered_test_target, stacking_ensemble.predict(
        credit_card_filtered_test_features))))
print('Recall score (filtered): {:.3f}'.format(
    recall_score(credit_card_filtered_test_target, stacking_ensemble.predict(
        credit_card_filtered_test_features))))

# Diversify the stacking ensemble to improve its performance
stacking_ensemble_diversified = Stacking(learner_levels = [[
    base_learners_diversified[l][1] for l in range(len(base_learners_diversified))], 
    [LogisticRegression()]])
# Original data
stacking_ensemble_diversified.fit(credit_card_train_features, 
                                  credit_card_train_target)
print(colored('\nAfter Diversification:', 'grey', attrs = ['bold']))
print('F1 score: {:.3f}'.format(
    f1_score(credit_card_test_target, stacking_ensemble_diversified.predict(
        credit_card_test_features))))
print('Recall score: {:.3f}'.format(
    recall_score(credit_card_test_target, stacking_ensemble_diversified.predict(
        credit_card_test_features))))

# Filtered data
stacking_ensemble_diversified.fit(credit_card_filtered_train_features, 
                                credit_card_filtered_train_target)
print('F1 score (filtered): {:.3f}'.format(f1_score(
    credit_card_filtered_test_target, stacking_ensemble_diversified.predict(
        credit_card_filtered_test_features))))
print('Recall score (filtered): {:.3f}'.format(recall_score(
    credit_card_filtered_test_target, stacking_ensemble_diversified.predict(
        credit_card_filtered_test_features))))

# Including an additional level of base learners
level1_base_learners = [DecisionTreeClassifier(max_depth = 2), LinearSVC()]
stacking_ensemble_added_level = Stacking(learner_levels = [[
    base_learners_diversified[l][1] for l in range(len(base_learners_diversified))], 
    level1_base_learners, [LogisticRegression()]])
# Original data
stacking_ensemble_added_level.fit(credit_card_train_features, 
                                  credit_card_train_target)
print(colored('\nAdding Another Level of Base Learners:', 'grey', 
              attrs = ['bold']))
print('F1 score: {:.3f}'.format(
    f1_score(credit_card_test_target, stacking_ensemble_added_level.predict(
        credit_card_test_features))))
f1_scores.append(('Stacking (Original)', f1_score(
    credit_card_test_target, stacking_ensemble_added_level.predict(
        credit_card_test_features))))
print('Recall score: {:.3f}'.format(
    recall_score(credit_card_test_target, stacking_ensemble_added_level.predict(
        credit_card_test_features))))
recall_scores.append(('Stacking (Original)', recall_score(
    credit_card_test_target, stacking_ensemble_added_level.predict(
        credit_card_test_features))))

# Filtered data
stacking_ensemble_added_level.fit(credit_card_filtered_train_features, 
                                  credit_card_filtered_train_target)
print('F1 score (filtered): {:.3f}'.format(f1_score(
    credit_card_filtered_test_target, stacking_ensemble_added_level.predict(
        credit_card_filtered_test_features))))
f1_scores.append(('Stacking (Filtered)', f1_score(
    credit_card_filtered_test_target, stacking_ensemble_added_level.predict(
        credit_card_filtered_test_features))))
print('Recall score (filtered): {:.3f}'.format(recall_score(
    credit_card_filtered_test_target, stacking_ensemble_added_level.predict(
        credit_card_filtered_test_features))))
recall_scores.append(('Stacking (Filtered)', recall_score(
    credit_card_filtered_test_target, stacking_ensemble_added_level.predict(
        credit_card_filtered_test_features))))

#### Bagging
bagging_ensemble = BaggingClassifier(n_estimators = 10, base_estimator = 
                                     DecisionTreeClassifier(max_depth = 8))
print(colored('\nBagging', 'cyan', attrs = ['bold']))

param_range_bagging = [5, 10, 15, 20, 25, 30]
training_validation_scores_bagging, testing_validation_scores_bagging = \
    validation_curve(
        bagging_ensemble, credit_card_train_features, credit_card_train_target,
        param_name = 'n_estimators', param_range = param_range_bagging, cv = 10,
        scoring = 'f1', n_jobs = -1)

training_mean_bagging = np.mean(training_validation_scores_bagging, axis = 1)
training_std_bagging = np.std(training_validation_scores_bagging, axis = 1)
testing_mean_bagging = np.mean(testing_validation_scores_bagging, axis = 1)
testing_std_bagging = np.std(testing_validation_scores_bagging, axis = 1)

plt.figure(7)
plt.title('Validation Curves (Original Data)')
# Standard deviation around each curve
plt.fill_between(param_range_bagging, 
                  training_mean_bagging - training_std_bagging,
                  training_mean_bagging + training_std_bagging, alpha = 0.2, 
                  color = 'C1')
plt.fill_between(param_range_bagging, testing_mean_bagging - testing_std_bagging,
                  testing_mean_bagging + testing_std_bagging, alpha = 0.2, 
                  color = 'C0')

# Mean (centre point) of each curve
plt.plot(param_range_bagging, training_mean_bagging, 'o-', color = 'C1', 
          label = 'Training Scores')
plt.plot(param_range_bagging, testing_mean_bagging, 'o-', color = 'C0',
          label = 'Cross-Validation Scores')
plt.xlabel('Size of Ensemble (Number of Decision Trees)')
plt.ylabel('F1 Score')
plt.legend(loc = 'best')

# Original data
bagging_ensemble.fit(credit_card_train_features, credit_card_train_target)
print('F1 score: {:.3f}'.format(
    f1_score(credit_card_test_target, bagging_ensemble.predict(
         credit_card_test_features))))
f1_scores.append(('Bagging (Original)', f1_score(
    credit_card_test_target, bagging_ensemble.predict(
        credit_card_test_features))))
print('Recall score: {:.3f}'.format(
    recall_score(credit_card_test_target, bagging_ensemble.predict(
        credit_card_test_features))))
recall_scores.append(('Bagging (Original)', recall_score(
    credit_card_test_target, bagging_ensemble.predict(
        credit_card_test_features))))

# Filtered data
bagging_ensemble.fit(credit_card_filtered_train_features, 
                     credit_card_filtered_train_target)
print('F1 score (filtered)): {:.3f}'.format(
    f1_score(credit_card_filtered_test_target, bagging_ensemble.predict(
        credit_card_filtered_test_features))))
f1_scores.append(('Bagging (Filtered)', f1_score(
    credit_card_filtered_test_target, bagging_ensemble.predict(
        credit_card_filtered_test_features))))
print('Recall score(filtered): {:.3f}'.format(
    recall_score(credit_card_filtered_test_target, bagging_ensemble.predict(
        credit_card_filtered_test_features))))
recall_scores.append(('Bagging (Filtered)', recall_score(
    credit_card_filtered_test_target, bagging_ensemble.predict(
        credit_card_filtered_test_features))))

#### Boosting (AdaBoost)
np.random.seed(123456)
boosting_ensemble = AdaBoostClassifier(n_estimators = 80,learning_rate = 1.4)
print(colored('\nBoosting', 'red', attrs = ['bold']))

param_range_boosting = [10, 40, 70, 100]
training_validation_scores_boosting, testing_validation_scores_boosting = \
    validation_curve(
        boosting_ensemble, credit_card_train_features, credit_card_train_target,
        param_name = 'n_estimators', param_range = param_range_boosting, cv = 10,
        scoring = 'f1', n_jobs = -1)

training_mean_boosting = np.mean(training_validation_scores_boosting, axis = 1)
training_std_boosting = np.std(training_validation_scores_boosting, axis = 1)
testing_mean_boosting = np.mean(testing_validation_scores_boosting, axis = 1)
testing_std_boosting = np.std(testing_validation_scores_boosting, axis = 1)

plt.figure(8)
plt.title('Validation Curves (Original Data)')
# Standard deviation around each curve
plt.fill_between(param_range_boosting, 
                  training_mean_boosting - training_std_boosting,
                  training_mean_boosting + training_std_boosting, alpha = 0.2, 
                  color = 'C1')
plt.fill_between(param_range_boosting, testing_mean_boosting - testing_std_boosting,
                  testing_mean_boosting + testing_std_boosting, alpha = 0.2, 
                  color = 'C0')

# Mean (centre point) of each curve
plt.plot(param_range_boosting, training_mean_boosting, 'o-', color = 'C1', 
          label = 'Training Scores')
plt.plot(param_range_boosting, testing_mean_boosting, 'o-', color = 'C0',
          label = 'Cross-Validation Scores')
plt.xlabel('Size of Ensemble')
plt.xticks(param_range_boosting)
plt.ylabel('F1 Score')
plt.legend(loc = 'best')

# Original data
boosting_ensemble.fit(credit_card_train_features, credit_card_train_target)
print('F1 score: {:.3f}'.format(
    f1_score(credit_card_test_target, boosting_ensemble.predict(
        credit_card_test_features))))
f1_scores.append(('Boosting (Original)', f1_score(
    credit_card_test_target, boosting_ensemble.predict(
        credit_card_test_features))))
print('Recall score: {:.3f}'.format(
    recall_score(credit_card_test_target, boosting_ensemble.predict(
        credit_card_test_features))))
recall_scores.append(('Boosting (Original)', recall_score(
    credit_card_test_target, boosting_ensemble.predict(
        credit_card_test_features))))

# Filtered data
boosting_ensemble.fit(credit_card_filtered_train_features, 
                      credit_card_filtered_train_target)
print('F1 score (filtered): {:.3f}'.format(
    f1_score(credit_card_filtered_test_target, boosting_ensemble.predict(
        credit_card_filtered_test_features))))
f1_scores.append(('Boosting (Filtered)', f1_score(
    credit_card_filtered_test_target, boosting_ensemble.predict(
        credit_card_filtered_test_features))))
print('Recall score (filtered): {:.3f}'.format(
    recall_score(credit_card_filtered_test_target, boosting_ensemble.predict(
        credit_card_filtered_test_features))))
recall_scores.append(('Boosting (Filtered)', recall_score(
    credit_card_filtered_test_target, boosting_ensemble.predict(
        credit_card_filtered_test_features))))

#### XGBoost
xgboost_ensemble = XGBClassifier(n_jobs = -1)
print(colored('\nUsing eXtreme Gradient Boost (XGBoost)', 'grey', 
              attrs = ['bold']))
# Original data
xgboost_ensemble.fit(credit_card_train_features, credit_card_train_target)
print('F1 score: {:.3f}'.format(
    f1_score(credit_card_test_target, xgboost_ensemble.predict(
        credit_card_test_features))))
f1_scores.append(('XGBoost (Original)', f1_score(
    credit_card_test_target, xgboost_ensemble.predict(
        credit_card_test_features))))
print('Recall score: {:.3f}'.format(
    recall_score(credit_card_test_target, xgboost_ensemble.predict(
        credit_card_test_features))))
recall_scores.append(('XGBoost (Original)', recall_score(
    credit_card_test_target, xgboost_ensemble.predict(
        credit_card_test_features))))

# Filtered data
xgboost_ensemble.fit(credit_card_filtered_train_features, 
                      credit_card_filtered_train_target)
print('F1 score (filtered): {:.3f}'.format(
    f1_score(credit_card_filtered_test_target, xgboost_ensemble.predict(
        credit_card_filtered_test_features))))
f1_scores.append(('XGBoost (Filtered)', f1_score(
    credit_card_filtered_test_target, xgboost_ensemble.predict(
        credit_card_filtered_test_features))))
print('Recall score (filtered): {:.3f}'.format(
    recall_score(credit_card_filtered_test_target, xgboost_ensemble.predict(
        credit_card_filtered_test_features))))
recall_scores.append(('XGBoost (Filtered)', recall_score(
    credit_card_filtered_test_target, xgboost_ensemble.predict(
        credit_card_filtered_test_features))))

#### RandomForest
randforest_ensemble = RandomForestClassifier(criterion = 'entropy', n_jobs = -1)
print(colored('\nRandom Forest', 'blue', attrs = ['bold']))

param_range_randforest = [10, 20, 30, 40, 50, 100, 150, 200, 300, 400, 500]
training_validation_scores_randforest, testing_validation_scores_randforest = \
    validation_curve(
        randforest_ensemble, credit_card_train_features, credit_card_train_target,
        param_name = 'n_estimators', param_range = param_range_randforest, 
        cv = 10,scoring = 'f1', n_jobs = -1)

training_mean_randforest = np.mean(training_validation_scores_randforest, axis = 1)
training_std_randforest = np.std(training_validation_scores_randforest, axis = 1)
testing_mean_randforest = np.mean(testing_validation_scores_randforest, axis = 1)
testing_std_randforest = np.std(testing_validation_scores_randforest, axis = 1)

plt.figure(9)
plt.title('Validation Curves (Original Data)')
# Standard deviation around each curve
plt.fill_between(param_range_randforest, 
                  training_mean_randforest - training_std_randforest,
                  training_mean_randforest + training_std_randforest, 
                  alpha = 0.2, color = 'C1')
plt.fill_between(param_range_randforest, 
                  testing_mean_randforest - testing_std_randforest,
                  testing_mean_randforest + testing_std_randforest, alpha = 0.2, 
                  color = 'C0')

# Mean (centre point) of each curve
plt.plot(param_range_randforest, training_mean_randforest, 'o-', color = 'C1', 
          label = 'Training Scores')
plt.plot(param_range_randforest, testing_mean_randforest, 'o-', color = 'C0',
          label = 'Cross-Validation Scores')
plt.xlabel('Size of Ensemble')
plt.xticks(param_range_randforest)
plt.ylabel('F1 Score')
plt.legend(loc = 'best')

# Original data
randforest_ensemble.fit(credit_card_train_features, credit_card_train_target)
print('F1 score: {:.3f}'.format(
    f1_score(credit_card_test_target, randforest_ensemble.predict(
        credit_card_test_features))))
f1_scores.append(('RandomForest (Original)', f1_score(
    credit_card_test_target, randforest_ensemble.predict(
        credit_card_test_features))))
print('Recall score: {:.3f}'.format(
    recall_score(credit_card_test_target, randforest_ensemble.predict(
        credit_card_test_features))))
recall_scores.append(('RandomForest (Original)', recall_score(
    credit_card_test_target, randforest_ensemble.predict(
        credit_card_test_features))))

# Filtered data
randforest_ensemble.fit(credit_card_filtered_train_features, 
                        credit_card_filtered_train_target)
print('F1 score (filtered): {:.3f}'.format(
    f1_score(credit_card_filtered_test_target, randforest_ensemble.predict(
        credit_card_filtered_test_features))))
f1_scores.append(('RandomForest (Filtered)', f1_score(
    credit_card_filtered_test_target, randforest_ensemble.predict(
        credit_card_filtered_test_features))))
print('Recall score (filtered): {:.3f}'.format(
    recall_score(credit_card_filtered_test_target, randforest_ensemble.predict(
        credit_card_filtered_test_features))))
recall_scores.append(('RandomForest (Filtered)', recall_score(
    credit_card_filtered_test_target, randforest_ensemble.predict(
        credit_card_filtered_test_features))))

#%% Comparative Analysis Of The Different Methods
#### F1 Scores
# Sort F1 scores in ascending order
f1_scores = np.sort(f1_scores, axis = 0)
ensembles_f1 = np.array([f1_scores[m][0] for m in range(len(f1_scores))], 
                        dtype = str)
f1_scores_separated = np.array([f1_scores[m][1] for m in range(len(f1_scores))],
                               dtype = np.float64)
# Plot the F1 scores
plt.figure(10)
plt.title('Comparison of F1 Scores')
plt.bar(ensembles_f1, f1_scores_separated)
plt.xticks(ensembles_f1, rotation = 'vertical')
plt.ylabel('F1 Scores')
plt.ylim([0.8, 0.9])
plt.subplots_adjust(bottom = 0.2)

#### Recall Scores
recall_scores = np.sort(recall_scores, axis = 0)
ensembles_recall = np.array(
    [recall_scores[n][0] for n in range(len(f1_scores))], dtype = str)
recall_scores_separated = np.array(
    [recall_scores[n][1] for n in range(len(f1_scores))],dtype = np.float64)

# Plot the recall scores
plt.figure(11)
plt.title('Comparison of Recall Scores')
plt.bar(ensembles_recall, recall_scores_separated)
plt.xticks(ensembles_f1, rotation = 'vertical')
plt.ylabel('Recall Scores')
plt.ylim([0.7, 0.9])
plt.subplots_adjust(bottom = 0.2)

stop = time.time()
print('\u2550'*100)
duration = stop - start
hours = divmod(divmod(duration, 60), 60)[0][0]
minutes = divmod(divmod(duration, 60), 60)[1][0]
seconds = divmod(divmod(duration, 60), 60)[1][1]
print(colored('Execution Duration: {:.2f}s ({:.1f}hrs, {:.1f}mins, {:.2f}s)'.format(
    duration, hours, minutes, seconds), 'red'))