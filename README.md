# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
In order to develop a model that simulates whether or not a consumer would sign up for a term deposit at the bank, this research looks at a marketing dataset that includes banking customers.

The AutoML(VotingEnsemble) algorithm produced the model with the best performance.

## Scikit-learn Pipeline
Data Preparation: Handle missing values and, if required, encode categorical variables as you preprocess and clean the bank marketing dataset.

Feature Engineering and Splitting: Divide the data into training and testing sets and carry out feature engineering as necessary.

Hyperparameter Sampling: Establish ranges for 'C' (regularization) and'max_iter' (maximum iterations) in a random hyperparameter sampler for logistic regression.

Early Stopping Policy: To improve computing efficiency, use BanditPolicy as the early-stopping policy to end underperforming runs.

Hyperdrive Configuration: Set up HyperDrive with the defined estimator, hyperparameter sampler, and early-stop strategy to automate the creation of models.

## AutoML
A Voting Ensemble is the best-performing model that AutoML can create. Predictions from several machine learning algorithms are combined in a voting ensemble, also referred to as a majority voting ensemble. By utilizing the advantages of several models, it aims to improve accuracy and stability. This is especially helpful when different models perform better with different kinds of input data.
## Pipeline comparison
Scikit-learn Pipeline - Accuracy - 91.56% AutoML - Accuracy - 91.75%
We can clearly state that AutoML outperformed the Scikit-learn pipeline, as it uses various models with n number of parameters to obtain the best suited model for the dataset

## Future work
With more resources and computational time the AutoML pipeline might be further enhanced. 
Additionally, obtaining additional training data would aid in the development of the model for this use-case.

