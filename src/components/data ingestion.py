import pandas as pd
import numpy as np
import sys
import os
from src.logger import logging
from src.exception import CustomException

from src.components import data_transformation
from dataclasses import dataclass
@dataclass

class DataIngestionConfig:
      train_file_paths = os.path.join(os.getcwd(),'original_data','train.csv')
      test_file_paths  = os.path.join(os.getcwd(),'original_data','test.csv')

class DataIngestion:
      def __init__(self):
            self.file_paths = DataIngestionConfig()

      def initiate_data_ingestion(self):
            logging.info("Data Ingestion Started")
            try:
                
                trainset = pd.read_csv(self.file_paths.train_file_paths)
                testset  = pd.read_csv(self.file_paths.test_file_paths)
                logging.info("Train & Test dataset ready for Data Ingestion Purposed")
                
                scaled_trainset = data_transformation.DataTransformation(trainset)
                scaled_testset  = data_transformation.DataTransformation(testset)
                logging.info("Data Transformation initiated to fix missing values & performed Scaling Operations")                
                
                scaled_train_dataset = scaled_trainset.initiate_data_transformation()
                scaled_train_dataset['Transported'] = trainset['Transported'].values

                logging.info("Scaling operation is performed on Train dataset")

                scaled_test_dataset  = scaled_testset.initiate_data_transformation()
                logging.info("Scaling operation is performed on Test dataset")   
                
                scaled_train_dataset.to_csv(os.path.join(os.getcwd(),'model_artifacts','scale_dataset','scaled_trainset.csv'),
                                            index=False)
                logging.info("Scaled Train dataset preserve inside model artifacts")

                scaled_test_dataset.to_csv(os.path.join(os.getcwd(),'model_artifacts','scale_dataset','scaled_testset.csv'),
                                           index=False)
                logging.info("Scaled Test dataset preserve inside model artifacts") 

            except Exception as e:
                  raise CustomException(e,sys)
            
if __name__=='__main__':
      objects = DataIngestion()
      objects.initiate_data_ingestion()