from src.logger import logging
from src.exception import CustomException
import warnings
warnings.filterwarnings("ignore")

from src.utils import Save_Object
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier

import pandas as pd
import numpy as np
import sys
import os

@dataclass

class DataTrainingConfig:
      trainset_file_path = os.path.join(os.getcwd(),"model_artifacts","train_test_split","trainset.csv")
      #testset_file_path = os.path.join(os.getcwd(),"artifacts","test_dataset.csv") 

class DataTraininPipeline:
      def __init__(self):
            self.data_training_config = DataTrainingConfig()

      def initiate_data_training_pipeline(self):
           logging.info("Data Training Pipeline Initiated") 
           try:
                trainset =  pd.read_csv(self.data_training_config.trainset_file_path)
                logging.info("Training Data available in artifacts")

                # testset  =  pd.read_csv(self.data_training_config.testset_file_path)
                # logging.info("Testing Data available in artifacts")

                X_train,y_train = trainset.iloc[:,:-1],trainset.iloc[:,-1]
                #X_test, y_test  = testset.iloc[:,:-1], testset.iloc[:,-1]

                logging.info("Dataset are split into Train Test")

                models ={
                         "RandomForest Classification":RandomForestClassifier(),
                         "DecisionTree Classification":DecisionTreeClassifier(),
                         "ExtraTree Classification":ExtraTreeClassifier(),
                         "KNN Classification":KNeighborsClassifier(),
                         "Logistic Regression":LogisticRegression(),
                         "Naive Bayes":MultinomialNB(),
                         "AdaBoot Classification":AdaBoostClassifier(),
                         "HistGradientBoostingClassifier":HistGradientBoostingClassifier()
                         }
                
                parameters ={}

                for model_name, model in models.items():
                      model.fit(X_train,y_train)
                      Save_Object(model,model_name)

                logging.info("Machine Learning Model Artifacts Stored in Buckets")  

           except Exception as e:
                  raise CustomException(e,sys)
           
if __name__=="__main__":
      obj = DataTraininPipeline()
      obj.initiate_data_training_pipeline()