# Starbucks Capstone Project Overview 
# Udacity Machine Learning Engineer Nanodegree

## Project Overview:
Starbucks is one of the most well-known companies in the world: a coffeehouse chain with more than 30 thousand stores all over the world. It strives to give their customers always the best service and the best experience. As a side feature, Starbucks offers their free app to make orders online, predict the waiting time and receive special offers. This app also offers promotions for bonus points to these users. The promotional offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). This project is focused on tailoring the customer behavior and responses while using the Starbucks mobile app. To avoid churn rate and trigger users to buy Starbucks products, it is important to know which offer should be sent to specific users.

To study about application of machine learning to predict customer churn, I used the reference:
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3835039.


## Datasets and Inputs:
There are 3 available data sources as mentioned below: 
1.	The first one is portfolio: it contains list of all available offers to propose to the customer. Each offer can be a discount, a BOGO (Buy One Get One) or Informational (no real offer), and we’ve got the details about discount, reward, and duration of the offer. 
2.	The next data source is profile, the list of all customers that interacted with the app. For each profile, the dataset contains some personal information like gender, age, and income. 
3.	Finally, there is the transcript dataset: it has the list of all actions on the app relative to special offers, plus all the customer’s transactions. For each record, we’ve got a dictionary of metadata, like offer_id and amount spent.

Here is the schema and explanation of each variable in the files:

portfolio.json: Offers sent during 30-day test period (10 offers x 6 fields)

- reward: (numeric) money awarded for the amount spent
- channels: (list) web, email, mobile, social
- difficulty: (numeric) money required to be spent to receive reward
- duration: (numeric) time for offer to be open, in days
- offer_type: (string) bogo, discount, informational
- id: (string/hash)

profile.json: Rewards program users (17000 users x 5 fields)

- gender: (categorical) M, F, O, or null
- age: (numeric) missing value encoded as 118
- id: (string/hash)
- became_member_on: (date) format YYYYMMDD
- income: (numeric)

transcript.json: Event log (306648 events x 4 fields)

- person: (string/hash)
- event: (string) offer received, offer viewed, transaction, offer completed
- value: (dictionary) different values depending on event type
    -  offer id: (string/hash) not associated with any "transaction"

    - amount: (numeric) money spent in "transaction"
    - reward: (numeric) money gained from "offer completed"
- time: (numeric) hours after start of test

## Libraries and Packages used:

- pandas- to work with tabular data in dataframes;
- numpy- to manipulate arrays and data structures;
- seaborn and matplotlib- for plotting visualizations;
- sklearn - to build and develop model pipeline;
- imblearn - to apply SMOTE
- xgboost - for using the classification model XGBClassifier
- sagemaker - to interact with AWS.

The AWS Services used are:
-  AWS Sagemaker for performing classification model training by leveraging hyperparameter tuning functionality available within it.
- AWS S3 Bucket for saving dataset and model artifacts.

## Files:
The following are the files included in this submission:
1. **proposal.pdf** - A proposal which details about the project's domain background, a problem statement, the datasets and inputs used, a solution statement, a benchmark model, an evaluation metrics, an outline of the project design.
2. **data** - This folder contains three types of files (in JSON format) provided by Udacity/Starbucks
3. **project-report.pdf** - A project report which details about Project Overview, Problem Statement, Metrics, Data Exploration, Exploratory Visualization, Algorithms and Techniques, Benchmark, Model Pipeline: Data Preprocessing and Implementation, Refinement, Model Evaluation and Validation, Conclusion.
4. **Analysis.ipynb** - Used for performing Exploratory Data Analysis. **preprocessing_script.py** is the supporting python script used for preprocessing numerical and categorical features.
5. **ModelTraining_Deployment.ipynb** - Used for training XGBoost classification model and deploying endpoint in AWS Sagemaker. Supporting folders and files for executing this file are:

    -  hpo-xgboost: used for performing hyperparameter tuning which contains

        1. requirements.txt: all the dependencies defined
        2. hpo.py: hyperparameter tuning job script
        3. preprocessing_script.py: script used for preprocessing numerical and categorical features
    -  best-hpo-model-training: used for performing model training using the best obtained hyperparameters

        1. requirements.txt: all the dependencies defined
        2. model_training.py: script for performing XGBoost classification model training with best obtained hyperparameters
        3. preprocessing_script.py: script used for preprocessing numerical and categorical features
    -  endpoint-serving: used to deploy an endpoint

        1. requirements.txt: all the dependencies defined
        2. endpoint.py: script for performing predictions through deployed endpoint.
        3. preprocessing_script.py: script used for preprocessing numerical and categorical features

## References:
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3835039
- https://www.mygreatlearning.com/blog/xgboost-algorithm/
- https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/
- https://sagemaker-examples.readthedocs.io/en/latest/hyperparameter_tuning/xgboost_random_log/hpo_xgboost_random_log.html
- https://www.kaggle.com/lucabasa/understand-and-use-a-pipeline
- https://www.kaggle.com/stuartday274/job-change-predictions-using-a-pipeline

## Acknowledgements:
I would like to thank Udacity and Starbucks for providing the dataset.