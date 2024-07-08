# Smart Mule Detector

This project aims to develop a framework for Bank A to proactively monitor transactions done by customers through their Savings Accounts in order to identify potential money mules. Money mules exploit online banking facilities to defraud others by convincing them to transfer money to accounts opened by the mules. Using machine learning algorithms, this project predicts the probability that a given account is a mule based on various account-level attributes, demographic information, transaction history, and other relevant factors.

## Table of Contents

- [Introduction](#introduction)
- [Data Description](#data-description)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Preprocessing](#preprocessing)
- [Feature Selection](#feature-selection)
- [Dimensionality Reduction](#dimensionality-reduction)
- [Model Selection and Training](#model-selection-and-training)
- [Metrics Used for Model Selection](#metrics-used-for-model-selection)
- [Model Analysis](#model-analysis)
- [Deep Learning Approach](#deep-learning-approach)
- [Ensemble of Top Performing Models](#ensemble-of-top-performing-models)
- [Results on the Validation Dataset](#results-on-the-validation-dataset)

## Introduction

### Problem Statement and Objective
The objective of this project is to develop a machine learning framework to identify potential money mules based on account-level attributes, demographic information, transaction history, and other relevant factors. This will help Bank A to proactively monitor and prevent fraudulent activities.

## Data Description

The development data contains 100,000 savings account details with 178 feature columns. The validation data consists of 50,000 savings account details with the same set of input variables but without the target column.

- **Primary key**: The index of the accounts.
- **Target**: 0 or 1, where accounts identified as mules have target=1.
- **Account level attributes**: e.g., account opening date.
- **Demographic attributes**: e.g., income, occupation, city tier, etc.
- **Transaction history**: Includes credits/debits of specific types over a period.
- **Other attributes**: e.g., product holding with the Bank, app/web logins, etc.

## Exploratory Data Analysis (EDA)

EDA was performed on columns such as account opening date, country code, income, city tier, occupation, OS, and email domains.

### Key Observations:
- **Account Opening Date**: Mule accounts were relatively less on weekends and more in December.
- **Country Code**: All mule accounts were found to be from India.
- **Income**: Majority of mule accounts had an income of less than 5 lacs.
- **City Tier**: Majority were from rural areas.
- **Occupation**: Most were self-employed.
- **OS**: Almost all were Android users.
- **Email Domain**: Most used Gmail.

## Preprocessing

### Steps:
- **Handling Missing Values**: Columns with more than 95,000 empty values were dropped. Categorical data missing values were filled with mode, and non-categorical data missing values were filled with mean.
- **Dropping Unnecessary Columns**: Columns with a single unique value were dropped.
- **Encoding Categorical Variables**: Used label encoder for large number of unique values in categorical features.

## Feature Selection

Due to the imbalanced dataset, mutual information was used for feature selection, which can capture non-linear relationships and handle imbalance. Features with mutual information less than 0.001 were dropped.

## Dimensionality Reduction

Applied PCA (Principal Component Analysis) after scaling the data. Set cumulative explained variance to 0.90, resulting in 68 principal components.

## Model Selection and Training

### Models Used:
- **XGB Classifier**: High performance, prevents overfitting.
- **Random Forest Classifier**: Handles class imbalance well.
- **K-Nearest Neighbors (KNN)**: Might not perform well for large, unbalanced datasets.
- **Support Vector Machines (SVM)**: Effective in high-dimensional spaces, works well with imbalanced data.

## Metrics Used for Model Selection

- **Classification Report**: Includes precision, recall, F1-score, and support for each class.
- **Average Precision Score**: Measures area under the precision-recall curve.
- **Balanced Accuracy Score**: Provides a balanced assessment of model's ability to predict both majority and minority classes.
- **Cohen Kappa Score**: Measures inter-rater agreement.
- **Cross Validation Score**: Used Stratified K-fold for splitting into 5 folds.
- **ROC AUC Score**: Measures ability to discriminate between positive and negative classes.

## Model Analysis

After training various models, XGB Classifier performed the best on the training data. Hyperparameter tuning and class weights were used to address class imbalance and improve performance.

## Deep Learning Approach

### Deep Neural Network
Created a sequential model with multiple dense layers, dropout layers, and relu/sigmoid activations. Both undersampling and oversampling techniques were used to handle class imbalance.

## Ensemble of Top Performing Models

The ensemble model, combining XGB Classifier and Neural Network, performed well on the validation data, reducing false positives without affecting false negatives.

## Results on the Validation Dataset

The ensemble model identified 518 mules out of 50,000 accounts in the validation dataset.
